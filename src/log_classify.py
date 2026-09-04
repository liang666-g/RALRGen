"""
使用 Qwen (DashScope) 预测英文日志有用性，采用强制 JSON 结构化输出
"""
# coding: utf-8
import os
import json
import time
import argparse
from openai import OpenAI

try:
    import local_api_config as local_api
except ImportError:
    local_api = None

def _local_attr(name, default=""):
    if local_api is None:
        return default
    return getattr(local_api, name, default)


API_KEY = str(_local_attr("LOG_CLASSIFY_API_KEY") or os.environ.get("LOG_CLASSIFY_API_KEY", "")).strip()
BASE_URL = str(_local_attr("LOG_CLASSIFY_BASE_URL") or os.environ.get("LOG_CLASSIFY_BASE_URL", "https://api.deepseek.com")).strip()
MODEL_NAME = str(_local_attr("LOG_CLASSIFY_MODEL") or os.environ.get("LOG_CLASSIFY_MODEL", "deepseek-v4-pro")).strip()
BATCH_SIZE = 50

if not API_KEY:
    raise ValueError("未配置 LOG_CLASSIFY_API_KEY，请在 local_api_config.py 中填写，或设置环境变量 LOG_CLASSIFY_API_KEY")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = '''You are a professional data annotation expert in Software Engineering.
Your task is to perform binary classification on App version release notes, extract truly informative release notes that are valuable to developers and users, and filter out meaningless filler, boilerplate, and polite but uninformative text.

[Classification Criteria]

Label 1 (Informative / Useful):
Mentions a specific addition, removal, or modification of a feature or UI screen. Examples: Added dark mode support, Removed old payment gateway.
Mentions a specific bug-fix scenario. Examples: Fixed a crash when opening the camera, Resolved login timeout issue.
Mentions a specific performance optimization point. Examples: Reduced app startup time by 20%, Improved video rendering speed.

Label 0 (Non-informative / Useless):The release note lacks specific details and belongs to vague boilerplate, marketing language, or version placeholders.
Vague bug-fix or optimization statements. Examples: Bug fixes and performance improvements, Minor updates, General enhancements.
Meaningless polite text or filler. Examples: Thanks for using our app!, Update to get the best experience, We update the app regularly.
Pure version numbers or placeholders. Examples: Version 2.0.1, V3, Update.

[Mandatory Output Format]
You must output only one valid JSON object. Do not include any Markdown markers, such as JSON code fences. Do not include any additional explanation or greeting.
The JSON must strictly follow this structure:
{
"results": [
{"idx": 1, "label": 1},
{"idx": 2, "label": 0}
]
}
'''

def generate_user_prompt(batch):
    prompt = (
        "Please classify the following batch of App release notes. "
        "Strictly return the result in the required JSON format.\n\n"
        "[Input Release Notes]\n"
    )
    for idx, record in enumerate(batch, 1):
        parts = record.strip().split('-[split]-')
        content = parts[2] if len(parts) >= 3 else ""
        prompt += f"[Log{idx}]：content-“{content}”\n"
    prompt += "\n[Classify the release notes and output JSON only]：\n"
    return prompt


def batch_process(batch_records, batch_num):
    """批量处理函数：加入强健的 JSON 解析和容错机制"""
    user_prompt = generate_user_prompt(batch_records)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # 降低随机性，保证分类稳定性
            response_format={"type": "json_object"} # 强制要求模型输出 JSON
        )
        
        output = response.choices[0].message.content
        # 清理可能存在的 markdown 标记
        clean_output = output.strip().strip('```json').strip('```')
        
        parsed_data = json.loads(clean_output)
        
        # 将 JSON 结果映射到字典中，防止大模型打乱顺序
        label_map = {}
        for item in parsed_data.get("results", []):
            label_map[str(item.get("idx"))] = str(item.get("label"))
            
        # 强制对齐：严格按照传入的 batch 顺序和长度生成结果
        results = []
        for i in range(1, len(batch_records) + 1):
            # 如果大模型漏掉了某一条，默认归为 0（无用），保障流程不崩
            results.append(label_map.get(str(i), "0"))
            
        return results

    except json.JSONDecodeError:
        print(f"批次{batch_num} JSON解析错误，模型输出了非规范格式。")
        return None
    except Exception as e:
        print(f"批次{batch_num} API错误: {str(e)}")
        return None


def single_process(record):
    """单条记录的容错处理"""
    user_prompt = generate_user_prompt([record])
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    output = response.choices[0].message.content
    clean_output = output.strip().strip('```json').strip('```')
    
    try:
        parsed_data = json.loads(clean_output)
        return str(parsed_data["results"][0]["label"])
    except:
        return "0"

def format_elapsed_time(start, end):
    elapsed = end - start
    hours = int(elapsed // 3600)
    remaining = elapsed % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{hours:02d}小时{minutes:02d}分{seconds:02d}秒"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用 Qwen 对英文 App 日志进行二分类并过滤")
    parser.add_argument('-i', '--input_file', type=str, required=True, 
                        help='输入的原始日志文件路径')
    parser.add_argument('-o', '--output_file', type=str, required=True, 
                        help='输出的有效日志文件路径')
    parser.add_argument('--max_lines', type=int, default=10000, 
                        help='最多读取的日志行数 (默认: 10000)')
    
    args = parser.parse_args()

    input_file_path = args.input_file
    filtered_file_path = args.output_file
    max_lines = args.max_lines

    base_name, ext = os.path.splitext(filtered_file_path)
    predict_file_path = f"{base_name}_temp{ext}"

    os.makedirs(os.path.dirname(predict_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(filtered_file_path), exist_ok=True)

    print(f"正在读取文件: {input_file_path}")
    start_time = time.time()
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            records = [line.strip() for i, line in enumerate(f) if 0 <= i < max_lines]  
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_file_path}'")
        exit(1)

    total = len(records)
    if total == 0:
        print("输入文件为空或未读取到有效数据。")
        exit(0)
        
    print(f"共读取到 {total} 条日志，将使用 {MODEL_NAME} 进行预测...")
    success_count = 0

    # 写入临时文件
    with open(predict_file_path, 'w', encoding='utf-8') as f:
        for idx in range(0, len(records), BATCH_SIZE):
            batch = records[idx:idx + BATCH_SIZE]
            batch_num = (idx // BATCH_SIZE) + 1
            print(f"开始处理-批次[{batch_num}]")
            s_time = time.time()
            
            batch_results = batch_process(batch, batch_num)

            # 只要有返回值，JSON 逻辑已经保证了长度一定对齐
            if batch_results:
                f.write('\n'.join(batch_results) + '\n')
                success_count += len(batch)
                e_time = time.time()
                print(f"批次[{batch_num}]完成 ({len(batch)}条)，耗时{e_time - s_time:.2f}秒")
                continue

            # 批量完全失败时才降级逐条处理
            print(f"批次-[{batch_num}] 批量处理失败，开始逐条重试...")
            for i, record in enumerate(batch, 1):
                retry = 2
                while retry > 0:
                    try:
                        result = single_process(record)
                        f.write(result + '\n')
                        success_count += 1
                        break
                    except Exception as e:
                        retry -= 1
                        time.sleep(2)
                else:
                    f.write("0\n") # 彻底失败默认标0丢弃
                    print(f"批次-[{batch_num}] 记录-[{idx + i}] 彻底失败，记为0")

    print(f"\n分类预测完成，成功率/容错覆盖率: {success_count}/{total} ({success_count / total:.2%})")

    # 根据分类结果提取日志
    print("开始根据 JSON 分类结果提取有效日志...")
    informative_logs = []
    
    with open(predict_file_path, 'r', encoding='utf-8') as f_pred:
        predictions = [line.strip() for line in f_pred if line.strip()]

    # 现在的对齐极其强壮，极大概率是相等的
    if len(predictions) == len(records):
        for record, pred in zip(records, predictions):
            if pred == '1':
                informative_logs.append(record)
        
        with open(filtered_file_path, 'w', encoding='utf-8') as f_out:
            for log in informative_logs:
                f_out.write(log + '\n')
                
        print(f"提取完成！")
        print(f"总记录数: {len(records)}")
        print(f"有效日志 (信息性): {len(informative_logs)} 条")
        print(f"无效废话 (被拦截): {len(records) - len(informative_logs)} 条")
        print(f"清洗后的最终文件已保存至：{filtered_file_path}")

        try:
            if os.path.exists(predict_file_path):
                os.remove(predict_file_path)
        except:
            pass
    else:
        print(f"致命异常：生成结果行数({len(predictions)})与输入({len(records)})不等！")

    end_time = time.time()
    print(f"总耗时: {format_elapsed_time(start_time, end_time)}")
