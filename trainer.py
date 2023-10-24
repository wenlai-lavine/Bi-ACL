# coding=utf-8
import torch
import numpy as np
import logging

logging.getLogger('transformers.generation_utils').disabled = True


def eval_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu
    eval_step = int(data.eval_num / dataset_batch_size) + 1
    val_loss, token_sum = 0., 0.
    model.eval()
    eval_data_loader_iter = iter(data.eval_dataloader)
    with torch.no_grad():
        eval_batch = next(eval_data_loader_iter)
        if cuda_available:
            eval_inputs = eval_batch['input_ids'].cuda(device)
            eval_labels = eval_batch['constraints']
        one_val_loss = model.module.eval_loss(
            eval_inputs,
            eval_labels
        )
        one_val_loss = torch.sum(one_val_loss)
        val_loss += one_val_loss.item()
    model.train()
    return val_loss


def model_training(args, data, model, total_steps, print_every, save_every, ckpt_save_path, cuda_available, device):
    import os
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)

    max_save_num = 1

    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_loss, train_cl_loss, min_val_loss = 0., 0., 1e10
    train_ave_bleu = 0.

    print('Start Training:')
    model.train()
    number_of_saves = 0

    # train dataloader iterations
    train_data_loader_iter = iter(data.train_dataloader)

    val_loss_not_change = []

    while effective_batch_acm < total_steps:
        all_batch_step += 1
        try:
            train_batch = next(train_data_loader_iter)
        except StopIteration:
            train_data_loader_iter = iter(data.train_dataloader)
            train_batch = next(train_data_loader_iter)
        if cuda_available:
            train_batch['input_ids'] = train_batch['input_ids'].cuda(device)
            train_batch['constraints'] = train_batch['constraints']
        all_loss, fwd_loss, bak_loss, fwd_cl_loss, bkd_cl_loss = model(
            input_ids=train_batch['input_ids'],
            labels=train_batch['constraints']
        )

        loss = all_loss.mean()
        loss.backward()
        train_loss += loss.mean().item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # parameter update
        if all_batch_step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            print_valid, save_valid = True, True

        # print intermediate result
        if effective_batch_acm % print_every == 0 and print_valid:
            denominator = (effective_batch_acm - (number_of_saves * save_every)) * gradient_accumulation_steps
            one_train_loss = train_loss / denominator
            print('At training steps {}, training MLE loss is {}'.format(effective_batch_acm, one_train_loss))
            print_valid = False

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False
            one_train_loss = train_loss / (save_every * gradient_accumulation_steps)

            model.eval()
            one_val_loss = eval_model(args, model, data, cuda_available, device)
            model.train()

            print('At training steps {}, training MLE loss is {}, validation loss is {}'.format(effective_batch_acm, one_train_loss, one_val_loss))

            train_loss, train_cl_loss = 0., 0.

            if one_val_loss < min_val_loss:
                # in finetuning stage, we always save the model
                min_val_loss = min(one_val_loss, min_val_loss)
                print('Saving model...')
                one_val_ppl = np.exp(one_val_loss)
                one_val_ppl = round(one_val_ppl, 3)
                save_name = 'training_step_{}_train_mle_loss_{}_dev_loss_{}_dev_ppl_{}'.format(effective_batch_acm, round(one_train_loss, 5), round(one_val_loss,5), one_val_ppl)

                model_save_path = ckpt_save_path + '/' + save_name
                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)
                if cuda_available and torch.cuda.device_count() > 1:
                    if args.local_rank == 0:
                        model.module.save_model(model_save_path)
                    else:
                        continue
                else:
                    model.save_model(model_save_path)
                print('Model Saved!')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = ckpt_save_path
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('training_step'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        if args.local_rank == 0:
                            one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                            os.system('rm -r ' + one_folder_name)
                        else:
                            continue
                print('-----------------------------------')
            else:
                if args.local_rank == 0:
                    val_loss_not_change.append(one_val_loss)
        if len(val_loss_not_change) > 5:
            break
    return model