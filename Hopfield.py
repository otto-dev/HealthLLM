import os
from docx import Document
import re
import torch.nn as nn
from hflayers import Hopfield, HopfieldPooling, HopfieldLayer
import torch.nn.functional as F
import torch
from sentence_transformers import SentenceTransformer
import pickle


class HopfieldRetrievalModel(nn.Module):
    def __init__(self, beta=0.125, update_steps_max=3):
        # def __init__(self, beta=0.125):
        super(HopfieldRetrievalModel, self).__init__()
        self.hopfield = Hopfield(
            scaling=beta,
            update_steps_max=update_steps_max,
            update_steps_eps=1e-5,

            # do not project layer input
            state_pattern_as_static=True,
            stored_pattern_as_static=True,
            pattern_projection_as_static=True,

            # do not pre-process layer input
            normalize_stored_pattern=False,
            normalize_stored_pattern_affine=False,
            normalize_state_pattern=False,
            normalize_state_pattern_affine=False,
            normalize_pattern_projection=False,
            normalize_pattern_projection_affine=False,

            # do not post-process layer output
            disable_out_projection=True)

    def forward(self, memory, trg):
        memory = torch.unsqueeze(memory, 0)
        trg = torch.unsqueeze(trg, 0)
        output = self.hopfield((memory, trg, memory))
        output = output.squeeze(0)
        memories = memory.squeeze(0)
        # temp = torch.bmm(F.softmax(attn_output_weights_init, dim=-1), memory).squeeze(0)
        pair_list = F.normalize(output) @ F.normalize(memories).t()  # step1
        return pair_list


def read_external_knowledge(path):
    path = '/Users/jmy/Desktop/ai_for_health_final/exsit_knowledge/my_dict.pkl'
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    paragraph = []
    for i in loaded_data:
        paragraph.append(loaded_data[i])
    return paragraph


def read_reports(path):
    reports = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            filepath = os.path.join(path, filename)

            # Read the .docx file
            with open(filepath, 'r') as f:
                txt = f.read()
                reports.extend(txt.split('\n'))

    return reports


def retrieval_info(reports, path, k):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paragraphs = read_external_knowledge(path + '/exsit_knowledge')
    print(len(paragraphs))

    # sentence_embedding with paragraphs
    model = SentenceTransformer('all-mpnet-base-v2')
    # p_embeddings = []
    # for i in paragraphs:
    #     p_embeddings.append(model.encode(i))
    p_embeddings = model.encode(paragraphs)
    # sentence_embedding with reports
    report_embeddings = model.encode(reports)
    print('report', report_embeddings.shape)
    print('p_embedding', p_embeddings.shape)
    retrievaler = HopfieldRetrievalModel().to(device)
    result = retrievaler(torch.tensor(p_embeddings).to(device) * 100, torch.tensor(report_embeddings).to(device) * 100)
    input_ids = torch.topk(result, k, dim=1).indices

    # mask = ~(input_ids == input_ids[0]).any(dim=1)
    # input_ids = input_ids[mask]
    indices = input_ids[0]
    # indices = set()
    # for input_id in input_ids:
    #     for id in input_id:
    #         indices.add(id.item())
    knowledge = []
    for indice in indices:
        knowledge.append(paragraphs[indice])
    knowledge = [x for x in knowledge if x != '']
    return knowledge


if __name__ == '__main__':
    reports = read_reports(
        '/Users/chongzhang/PycharmProjects/ai_for_health_final/dataset_folder/health_report_{243}')  # 13452
    know = retrieval_info(reports, '/Users/chongzhang/PycharmProjects/ai_for_health_final/', 3)
    for i in know:
        print(i)
