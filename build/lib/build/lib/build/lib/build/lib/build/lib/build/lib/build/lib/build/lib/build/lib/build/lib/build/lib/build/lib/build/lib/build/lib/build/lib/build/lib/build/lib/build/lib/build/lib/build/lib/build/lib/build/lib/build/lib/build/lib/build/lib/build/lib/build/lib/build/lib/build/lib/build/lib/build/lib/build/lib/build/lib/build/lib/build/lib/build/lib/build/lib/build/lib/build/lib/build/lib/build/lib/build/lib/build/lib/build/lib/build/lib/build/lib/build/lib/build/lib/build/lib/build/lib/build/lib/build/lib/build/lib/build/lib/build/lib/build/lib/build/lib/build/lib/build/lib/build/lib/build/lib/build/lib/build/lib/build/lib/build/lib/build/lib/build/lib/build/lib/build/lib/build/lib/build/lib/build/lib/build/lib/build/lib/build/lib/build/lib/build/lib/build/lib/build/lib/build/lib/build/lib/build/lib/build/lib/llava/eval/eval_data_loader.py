import json
import os
import random

from PIL import Image
from torch.utils.data import Dataset

from llava.mm_utils import process_images




class COCODataSet(Dataset):
    def __init__(self, data_path, trans, model_config):
        self.data_path = data_path
        self.trans = trans

        img_files = os.listdir(self.data_path)
        random.shuffle(img_files)
        self.img_files = img_files
        self.model_config = model_config

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_file = self.img_files[index]
        img_id = int(img_file.split(".jpg")[0][-6:])

        image = Image.open(os.path.join(self.data_path, img_file)).convert("RGB")
        image_tensor = process_images([image], self.trans, self.model_config)[0]
        return {"img_id": img_id, "image": image_tensor}
    

class POPEDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []
        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            if label_list[i] == "no":
                label_list[i] = 0
            else:
                label_list[i] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label}


class POPEChatDataSet(Dataset):
    def __init__(self, pope_path, data_path, trans):
        self.pope_path = pope_path
        self.data_path = data_path
        self.trans = trans

        image_list, query_list, label_list = [], [], []

        for q in open(pope_path, "r"):
            line = json.loads(q)
            image_list.append(line["image"])
            query_list.append(line["text"])
            label_list.append(line["label"])

        for i in range(len(label_list)):
            for j in range(len(label_list[i])):
                if label_list[i][j] == "no":
                    label_list[i][j] = 0
                else:
                    label_list[i][j] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(label_list)

        self.image_list = image_list
        self.query_list = query_list
        self.label_list = label_list

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        label = self.label_list[index]

        return {"image": image, "query": query, "label": label}


class SYChatDataSet(Dataset):
    def __init__(self, trans):
        self.pope_path = '/mnt/data1/user/li_shuo/TDIUC/inner_dataset_pro_all.json'
        self.data_path = '/mnt/data1/user/li_shuo/TDIUC/Images/val2014'
        self.trans = trans

        image_list, query_list, true_option_list,sy_option_list = [], [], [], []
        with open(self.pope_path,'r') as f:
            chat_json = json.load(f)
        for chat in chat_json:
            # line = json.loads(q)
            question=chat['question']
            true_option=chat['true option']
            sy_option=chat['sy option']
            image_id=chat['image_id']
            solid_response=chat['solid_response']
            gentle_response=chat['gentle_response']
            suggest_response=chat['suggest_response']
            image_list.append(os.path.join(self.data_path,image_id+'.jpg'))
            query_list.append([question,random.choice([solid_response,gentle_response,suggest_response])])
            true_option_list.append(chat['true option'])
            sy_option_list.append(chat['sy option'])
            # label_list.append(line["label"])

        # for i in range(len(label_list)):
        #     for j in range(len(label_list[i])):
        #         if label_list[i][j] == "no":
        #             label_list[i][j] = 0
        #         else:
        #             label_list[i][j] = 1

        assert len(image_list) == len(query_list)
        assert len(image_list) == len(true_option_list)

        self.image_list = image_list
        self.query_list = query_list
        self.true_option_list = true_option_list
        self.sy_option_list = sy_option_list

    def __len__(self):
        return len(self.true_option_list)

    def __getitem__(self, index):
        image_path = os.path.join(self.data_path, self.image_list[index])
        raw_image = Image.open(image_path).convert("RGB")
        image = self.trans(raw_image)
        query = self.query_list[index]
        true_option=self.true_option_list[index]
        sy_option=self.sy_option_list[index]
        # label = self.label_list[index]

        return {"image": image, "query": query, "true option":true_option,"sy option":sy_option}