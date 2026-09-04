import pandas as pd
import re
from datetime import datetime
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import time
import argparse
'''
App信息预处理：提取日期-版本号-更新日志 V4
从csv文件提取数据 更新日志单元格提取每行数据，时间升序排序，去除重复日志，去除日志标题--2020-04-08-1.4.0-上传头像 除数字+、
每项使用-[split]-分隔
命令行处理参数
python log_process.py --input ./data/app_logs/log_source_file/Snapchat版本记录_2026-04-13.csv --output ./data/app_logs/logs/com.snapchat.android.txt

'''

# 输入数据为多个日志，需分开并优化


def parse_changelog_entries(changelog_text):
    """
    解析更新日志文本，提取多条更新记录，排除以"【"开头且以"】"结尾的条目

    参数:
    changelog_text: 原始更新日志文本

    返回:
    list: 清理后的更新日志条目列表
    """
    if not changelog_text or pd.isna(changelog_text) or str(changelog_text).strip() == '':
        return []

    changelog_text = str(changelog_text)

    lines = changelog_text.split('\n')

    if len(lines) == 1:
        lines = re.split(r';|\. |。', changelog_text)

    parsed_entries = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        line = re.sub(r'^[\s\+\-·•*◦▪▫●=]\s*', '', line)  # 去除行首的项目符号（包括后面的空格）
        line = re.sub(r'^\d+\.\s*', '', line)  # 1.  2.
        line = re.sub(r'^\d+\)\s*', '', line)  # 1)  2)
        line = re.sub(r'^\d+、\s*', '', line)  # 1、  2、
        line = re.sub(r'^\d+）\s*', '', line)  # 1）  2）
        line = re.sub(r'^\d+，\s*', '', line)  # 1，  2，
        line = re.sub(r'^[一二三四五六七八九十]+、\s*', '', line)  # 一、  二、
        line = re.sub(r'^[a-zA-Z]\.\s*', '', line)  # A.  B.
        line = re.sub(r'^[a-zA-Z]\)\s*', '', line)  # A)  B)
        line = re.sub(r'^[ivxlcdm]+\.\s*', '', line)  # i.  ii.
        line = re.sub(r'^[IVXLCDM]+\.\s*', '', line)  # I.  II.
        line = re.sub(r'^[a-zA-Z]-\s*', '', line)  # A-  B-
        line = re.sub(r'^=-\s*', '', line)  # =-  =-
        line = re.sub(r'-', '', line)  # 去除-替换为空格

        line = re.sub(r'<[^>]+>', '', line)

        line = re.sub(r'\s+', ' ', line)
        line = line.strip()

        if line.endswith('.') and not line.endswith('...'):
            line = line[:-1].strip()
        if line.endswith('。'):
            line = line[:-1].strip()
        if line.endswith(';'):
            line = line[:-1].strip()
        if line.endswith('；'):
            line = line[:-1].strip()
        if line.endswith('~'):
            line = line[:-1].strip()
        if line.endswith('！'):
            line = line[:-1].strip()
        if line.endswith('!'):
            line = line[:-1].strip()

        if not line or len(line) < 2:
            continue

        if line.lower() in ['null', 'none', 'n/a', '无', '暂无',
                            'nothing', 'no update', 'bug fix', 'Performance improvements'
                            'no updates', 'no changes', 'no new features']:
            continue

        if line.startswith('【') and line.endswith('】'):
            continue

        if line.startswith('##'):
            continue

        parsed_entries.append(line)

    return parsed_entries


def normalize_date(date_str):
    """
    标准化日期格式为 YYYY-MM-DD

    参数:
    date_str: 原始日期字符串

    返回:
    str: 标准化后的日期字符串 (YYYY-MM-DD)
    """
    if not date_str or pd.isna(date_str):
        return None

    date_str = str(date_str).strip()

    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # 若需求英文日期的格式，在此处添加，如13 Jun 2025
    date_patterns = [
        r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # 2025/6/13, 2025-6-13
        r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # 2025年6月13日
        r'(\d{4})\.(\d{1,2})\.(\d{1,2})',  # 2025.6.13
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',  # 13/6/2025, 13-6-2025
        r'(\d{1,2})\.(\d{1,2})\.(\d{4})',  # 13.6.2025
    ]

    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            groups = match.groups()

            if len(groups[0]) == 4:
                year, month, day = groups[0], groups[1], groups[2]
            else:
                day, month, year = groups[0], groups[1], groups[2]

            month = month.zfill(2)
            day = day.zfill(2)

            try:
                datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")
                return f"{year}-{month}-{day}"
            except ValueError:
                continue

    return date_str


def process_app_info_csv(input_file, output_file):
    """
    处理CSV文件，提取版本信息并写入txt文件，先去重后排除与之前条目余弦相似度达到0.9的更新日志

    参数:
    input_file: 输入CSV文件路径
    output_file: 输出txt文件路径
    """
    try:
        df = pd.read_csv(input_file, encoding='utf-8', quoting=1, escapechar='\\')
        print(f"成功读取CSV文件，共{len(df)}行数据")
    except Exception as e:
        print(f"UTF-8编码读取失败，尝试GBK编码: {e}")
        try:
            df = pd.read_csv(input_file, encoding='gbk', quoting=1, escapechar='\\')
            print(f"成功读取CSV文件，共{len(df)}行数据")
        except Exception as e2:
            print(f"读取CSV文件失败: {e2}")
            return

    required_columns = ['版本更新日期', '版本号', '更新日志']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"缺少必要的列: {missing_columns}")
        print(f"实际列名: {list(df.columns)}")
        return

    all_entries = []
    skipped_rows = 0
    filtered_by_date = 0

    start_date = datetime.strptime("2015-01-01", "%Y-%m-%d")
    end_date = datetime.strptime("2020-12-31", "%Y-%m-%d")

    for index, row in df.iterrows():
        try:
            update_date = row['版本更新日期']
            version = row['版本号']
            changelog = row['更新日志']

            if pd.isna(update_date) or pd.isna(version) or pd.isna(changelog):
                skipped_rows += 1
                continue

            update_date = str(update_date).strip()
            version = str(version).strip()
            changelog = str(changelog).strip()

            normalized_date = normalize_date(update_date)
            if not normalized_date:
                print(f"第{index + 1}行日期格式无效: {update_date}")
                skipped_rows += 1
                continue

            # 只保留 2015-01-01 到 2020-12-31 的数据
            # try:
            #     current_date = datetime.strptime(normalized_date, "%Y-%m-%d")
            # except ValueError:
            #     print(f"第{index + 1}行日期无法解析为标准格式: {normalized_date}")
            #     skipped_rows += 1
            #     continue

            # if current_date < start_date or current_date > end_date:
            #     filtered_by_date += 1
            #     continue

            if not changelog or changelog == '' or changelog == 'nan':
                skipped_rows += 1
                continue

            changelog_entries = parse_changelog_entries(changelog)

            if not changelog_entries:
                skipped_rows += 1
                continue

            for changelog_item in changelog_entries:
                entry = {
                    'date': normalized_date,
                    'version': version,
                    'changelog': changelog_item,
                    'sort_date': datetime.strptime(normalized_date, "%Y-%m-%d")
                }
                all_entries.append(entry)

        except Exception as e:
            print(f"处理第{index + 1}行时出错: {e}")
            skipped_rows += 1
            continue

    print(f"解析后共有{len(all_entries)}条日志记录，不做精确去重和相似度去重")
    final_entries = sorted(all_entries, key=lambda x: x['sort_date'])

    output_lines = []
    for entry in final_entries:
        # 文件格式 date-[split]-version-[split]-content-[split]-label
        output_line = f"{entry['date']}-[split]-{entry['version']}-[split]-{entry['changelog']}"
        output_lines.append(output_line)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in output_lines:
                f.write(line + '\n')

        print(f"跳过{skipped_rows}行无效数据")
        # print(f"因不在2015-2020年范围内跳过{filtered_by_date}行数据")
        print(f"成功处理{len(all_entries)}条原始记录")
        print(f"最终写入{len(final_entries)}条日志记录")
        print(f"结果已按日期升序写入: {output_file}")

    except Exception as e:
        print(f"写入文件失败: {e}")


def format_elapsed_time(start, end):
    elapsed = end - start
    hours = int(elapsed // 3600)
    remaining = elapsed % 3600
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{hours:02d}小时{minutes:02d}分{seconds:02d}秒"


# 使用示例
if __name__ == "__main__":

    # python 03_log_process_cn.py --input_path ../data/log/kitchen/kitchen.csv --output_path ../data/log/kitchen/kitchen_log.txt
    parser = argparse.ArgumentParser(description="pre-process review data")
    parser.add_argument("--input", type=str, default="./data/app_logs/log source file/Uber - Request a ride版本记录_2026-04-13.csv", help="Path of input xlsx file")
    parser.add_argument("--output", type=str, default="./data/app_logs/logs/com.ubercab.txt", help="Path of output txt file")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("开始处理数据...")

    start_time = time.time()
    process_app_info_csv(args.input, args.output)
    end_time = time.time()

    print("\n" + "=" * 60)
    print("处理完成!")
    print("共耗时：", format_elapsed_time(start_time, end_time))
