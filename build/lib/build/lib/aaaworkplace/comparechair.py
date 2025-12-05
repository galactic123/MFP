import json

# 读取JSON文件
with open('/opt/project/n/mm_school/fudan/project/user/sjj/newfuse/newfuse/LLaVA-main/aaaworkplace/chairs_023.json', 'r') as f1, open('/opt/project/n/mm_school/fudan/project/user/sjj/newfuse/newfuse/LLaVA-main/aaaworkplace/result_oris.json', 'r') as f2:
    json1 = json.load(f1)
    json2 = json.load(f2)

# 创建一个字典以便快速查找json2中的条目
json2_dict = {item['image_id']: item for item in json2['sentences']}

# 存储符合条件的条目
results = []

# 遍历json1中的条目
for item in json1['sentences']:
    image_id = item['image_id']
    metrics = item['metrics']
    if 'CHAIRs' not in metrics or 'F1' not in metrics:
        continue
    # 检查条件
    if metrics['CHAIRs'] == 0 and metrics['F1'] == 1.0:
        if image_id in json2_dict:
            json2_metrics = json2_dict[image_id]['metrics']
            if json2_metrics['CHAIRs'] > 0:
                results.append({
                    'image_id': image_id,
                    'json1_metrics': metrics,
                    'json2_metrics': json2_metrics
                })

# 将结果保存到新的JSON文件
with open('/opt/project/n/mm_school/fudan/project/user/sjj/newfuse/newfuse/LLaVA-main/aaaworkplace/results.json', 'w') as outfile:
    json.dump(results, outfile, indent=2)

print("结果已保存到 results.json")
