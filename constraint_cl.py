#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn.functional as F
import torch
import torch.nn as nn

from transformers import (
    MBartTokenizer,
    MBartTokenizerFast,
    M2M100Tokenizer,
    PhrasalConstraint
)


class Cons_CL(nn.Module):
    def __init__(self, model_name, data, user_defined_model=None, user_defined_tokenizer=None, special_token_list=[]):
        super(Cons_CL, self).__init__()
        '''
            user_defined_model: whether user would like to use self-defined model
            user_defined_tokenizer: whether user would like to use self-defined tokenizer

            if user_defined_model and user_defined_tokenizer 
        '''
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        if user_defined_model is None:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            print('Use user defined model ... ...')
            self.model = user_defined_model

        self.tokenizer = data.tokenizer
        if len(special_token_list) > 0:
            print('Original vocabulary size is {}'.format(len(self.tokenizer)))
            print('Adding special tokens...')
            self.tokenizer.add_tokens(special_token_list)
            print('Special token added.')
            print('Resizing language model embeddings...')
            self.model.resize_token_embeddings(len(self.tokenizer))
            print('Language model embeddings resized.')
        self.vocab_size = len(self.tokenizer)
        print('The vocabulary size of the language model is {}'.format(len(self.tokenizer)))
        self.embed_dim = self.model.config.hidden_size

        self.projection = nn.Sequential(nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size), nn.ReLU())

        # set decoder_start_token_id
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, M2M100Tokenizer)):
            if isinstance(self.tokenizer, (MBartTokenizer, MBartTokenizerFast, M2M100Tokenizer)):
                self.model.config.decoder_start_token_id = self.tokenizer.lang_code_to_id[data.target_lang]
            else:
                self.model.config.decoder_start_token_id = self.tokenizer.convert_tokens_to_ids(data.target_lang)

        self.contrastive_lambda = 0.5
        self.temperature = 0.1

    def shift_tokens_right(self, input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
        """
        Shift input ids one token to the right.
        """
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = decoder_start_token_id

        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        return shifted_input_ids

    def gen_constraint_label(self, input_ids, constraint):
        self.model.eval()
        contrs_outputs = self.model.generate(
            input_ids=input_ids.unsqueeze(0),
            force_words_ids=constraint,
            max_length=64,
            forced_bos_token_id=self.tokenizer.get_lang_id(self.tokenizer.tgt_lang))
        return contrs_outputs

    def get_constraint_batch_labels(self, input_ids, labels):
        batch_size = input_ids.size(0)
        res_text = []
        for i in range(batch_size):
            constraint_output = self.gen_constraint_label(input_ids[i], labels[i])
            res_text.append(constraint_output)
        return res_text

    def pad_labels(self, labels):
        pad_labels = []
        for i in range(len(labels)):
            cons_labes_list = labels[i].tolist()[0][1:]
            if cons_labes_list[0] == self.tokenizer.get_lang_id(self.tokenizer.tgt_lang):
                cons_labes_list = cons_labes_list + [self.tokenizer.pad_token_id] + [-100] * (
                            128 - 1 - len(cons_labes_list))
            else:
                cons_labes_list = [self.tokenizer.get_lang_id(self.tokenizer.tgt_lang)] + cons_labes_list + [
                    self.tokenizer.pad_token_id] + [-100] * (128 - 1 - 1 - len(cons_labes_list))
            pad_labels.append(cons_labes_list)
        return pad_labels

    def change_pad(self, labels):
        # -100 --> 1
        pad_labels = []
        for i in range(labels.size(0)):
            tmp_res = []
            cons_labes_list = labels.tolist()[i]
            for i in cons_labes_list:
                if i != -100:
                    tmp_res.append(i)
                else:
                    tmp_res.append(self.tokenizer.pad_token_id)
            pad_labels.append(tmp_res)
        return pad_labels

    def change_pad_to_ignore_loss(self, labels):
        # 1 --> -100
        pad_labels = []
        for i in range(labels.size(0)):
            tmp_res = []
            cons_labes_list = labels.tolist()[i]
            for i in cons_labes_list:
                if i != 1:
                    tmp_res.append(i)
                else:
                    tmp_res.append(-100)
            pad_labels.append(tmp_res)
        return pad_labels

    def get_mask_from_input_ids(self, input_ids):
        batch_size, seq_len = input_ids.shape
        input_ids_list = input_ids.tolist()
        res_mask_list = []
        for batch in range(batch_size):
            tmp_mask_list = []
            for ids in input_ids_list[batch]:
                if ids != self.tokenizer.pad_token_id or ids != -100:
                    tmp_mask_list.append(1)
                else:
                    tmp_mask_list.append(0)
            res_mask_list.append(tmp_mask_list)
        res_mask_tensor = torch.tensor(res_mask_list).cuda()
        return res_mask_tensor

    def forward(self, input_ids, labels):
        constraint_labels = self.get_constraint_batch_labels(input_ids, labels)
        self.model.train()
        constraint_labels_pad = torch.tensor(self.pad_labels(constraint_labels)).cuda()
        forward_outputs = self.model(input_ids=input_ids, labels=constraint_labels_pad)
        forward_loss = forward_outputs.loss
        constraint_labels_change_pad = torch.tensor(self.change_pad(constraint_labels_pad)).cuda()
        input_ids_change_pad = torch.tensor(self.change_pad_to_ignore_loss(input_ids)).cuda()
        backward_loss = self.model(input_ids=constraint_labels_change_pad, labels=input_ids_change_pad)
        backward_loss = backward_loss.loss
        fwd_cl_loss = self.contrastive_loss(input_ids, constraint_labels_pad)
        bkd_cl_loss = self.contrastive_loss(constraint_labels_change_pad, input_ids_change_pad)

        all_loss = forward_loss + backward_loss + fwd_cl_loss + bkd_cl_loss

        return all_loss, forward_loss, backward_loss, fwd_cl_loss, bkd_cl_loss

    def contrastive_loss(self, input_ids, labels):
        encoder = self.model.get_encoder()
        decoder = self.model.get_decoder()
        attention_mask = self.get_mask_from_input_ids(input_ids)
        encoder_outputs = encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = encoder_outputs[0]
        decoder_input_ids = self.shift_tokens_right(labels, self.model.config.pad_token_id,
                                                    self.model.config.decoder_start_token_id)
        decoder_attention_mask = self.get_mask_from_input_ids(decoder_input_ids)
        decoder_outputs = decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask,
                                  encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask)
        sequence_output = decoder_outputs[0]

        sequence_output = sequence_output * (self.model.config.d_model ** -0.5)
        lm_logits = self.model.lm_head(sequence_output)
        proj_enc_h = self.projection(hidden_states)
        proj_dec_h = self.projection(sequence_output)
        avg_doc = self.avg_pool(proj_enc_h, attention_mask)
        avg_abs = self.avg_pool(proj_dec_h, decoder_attention_mask)
        cos = nn.CosineSimilarity(dim=-1)
        cont_crit = nn.CrossEntropyLoss()
        sim_matrix = cos(avg_doc.unsqueeze(1), avg_abs.unsqueeze(0))
        perturbed_dec = self.generate_adv(sequence_output, labels)  # [n,b,t,d] or [b,t,d]
        batch_size = input_ids.size(0)

        proj_pert_dec_h = self.projection(perturbed_dec)
        avg_pert = self.avg_pool(proj_pert_dec_h, decoder_attention_mask)
        adv_sim = cos(avg_doc, avg_pert).unsqueeze(1)  # [b,1]
        pos_dec_hidden = self.generate_cont_adv(hidden_states, attention_mask,
                                                sequence_output, decoder_attention_mask,
                                                lm_logits,
                                                self.tau, self.pos_eps)
        avg_pos_dec = self.avg_pool(self.projection(pos_dec_hidden),
                                    decoder_attention_mask)

        pos_sim = cos(avg_doc, avg_pos_dec).unsqueeze(-1)  # [b,1]
        logits = torch.cat([sim_matrix, adv_sim], 1) / self.tau

        identity = torch.eye(batch_size, device=input_ids.device)
        pos_sim = identity * pos_sim
        neg_sim = sim_matrix.masked_fill(identity == 1, 0)
        new_sim_matrix = pos_sim + neg_sim
        new_logits = torch.cat([new_sim_matrix, adv_sim], 1)

        labels = torch.arange(batch_size,
                              device=input_ids.device)

        cont_loss = cont_crit(logits, labels)
        new_cont_loss = cont_crit(new_logits, labels)

        cont_loss = 0.5 * (cont_loss + new_cont_loss)

        return cont_loss

    def generate_adv(self, dec_hiddens, lm_labels):
        dec_hiddens = dec_hiddens.detach()

        dec_hiddens.requires_grad = True

        lm_logits = self.model.lm_head(dec_hiddens)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        loss = criterion(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
        loss.backward()
        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturbed_dec = dec_hiddens + self.neg_eps * dec_grad.detach()
        perturbed_dec = perturbed_dec  # [b,t,d]

        self.zero_grad()

        return perturbed_dec

    def generate_cont_adv(self, enc_hiddens, enc_mask,
                          dec_hiddens, dec_mask, lm_logits,
                          tau, eps):
        enc_hiddens = enc_hiddens.detach()
        dec_hiddens = dec_hiddens.detach()
        lm_logits = lm_logits.detach()
        dec_hiddens.requires_grad = True

        avg_enc = self.avg_pool(self.projection(enc_hiddens), enc_mask)

        avg_dec = self.avg_pool(self.projection(dec_hiddens), dec_mask)

        cos = nn.CosineSimilarity(dim=-1)
        logits = cos(avg_enc.unsqueeze(1), avg_dec.unsqueeze(0)) / tau

        cont_crit = nn.CrossEntropyLoss()
        labels = torch.arange(avg_enc.size(0), device=enc_hiddens.device)
        loss = cont_crit(logits, labels)
        loss.backward()

        dec_grad = dec_hiddens.grad.detach()
        l2_norm = torch.norm(dec_grad, dim=-1)
        dec_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = dec_hiddens + eps * dec_grad
        perturb_dec_hidden = perturb_dec_hidden.detach()
        perturb_dec_hidden.requires_grad = True
        perturb_logits = self.model.lm_head(perturb_dec_hidden)

        true_probs = F.softmax(lm_logits, -1)
        true_probs = true_probs * dec_mask.unsqueeze(-1).float()

        perturb_log_probs = F.log_softmax(perturb_logits, -1)
        kl_crit = nn.KLDivLoss(reduction="sum")
        vocab_size = lm_logits.size(-1)

        kl = kl_crit(perturb_log_probs.view(-1, vocab_size),
                     true_probs.view(-1, vocab_size))
        kl = kl / torch.sum(dec_mask).float()
        kl.backward()

        kl_grad = perturb_dec_hidden.grad.detach()

        l2_norm = torch.norm(kl_grad, dim=-1)

        kl_grad /= (l2_norm.unsqueeze(-1) + 1e-12)

        perturb_dec_hidden = perturb_dec_hidden - eps * kl_grad

        return perturb_dec_hidden

    def avg_pool(self, hidden_states, mask):
        length = torch.sum(mask, 1, keepdim=True).float()
        mask = mask.unsqueeze(2)
        hidden = hidden_states.masked_fill(mask == 0, 0.0)
        avg_hidden = torch.sum(hidden, 1) / length

        return avg_hidden

    def eval_loss(self, input_ids, labels):
        self.model.eval()
        constraint_labels = self.get_constraint_batch_labels(input_ids, labels)
        constraint_labels_pad = torch.tensor(self.pad_labels(constraint_labels)).cuda()
        forward_outputs = self.model(input_ids=input_ids, labels=constraint_labels_pad)
        forward_loss = forward_outputs.loss
        constraint_labels_change_pad = torch.tensor(self.change_pad(constraint_labels_pad)).cuda()
        input_ids_change_pad = torch.tensor(self.change_pad_to_ignore_loss(input_ids)).cuda()
        backward_loss = self.model(input_ids=constraint_labels_change_pad, labels=input_ids_change_pad)
        backward_loss = backward_loss.loss
        all_loss = forward_loss + backward_loss
        return all_loss

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else:  # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)
        # save model
        self.model.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)