import os
import random

def run_random_baseline():
    # ================= 1. 配置文件路径 =================
    train_file = "./data/train.txt"
    test_file = "./data/test.txt"
    
    # 输出目录 (自动创建 Random 文件夹)
    out_dir = "./contrast_model_response/Random"
    os.makedirs(out_dir, exist_ok=True)
    
    out_pred_path = os.path.join(out_dir, "pred.txt")
    out_tar_path = os.path.join(out_dir, "tar.txt")

    # ================= 2. 提取训练集回复池 =================
    print(f"📖 正在读取训练集，构建候选回复池: {train_file}")
    train_replies = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            terms = line.strip().split("-[split]-")
            if len(terms) >= 6:
                # 索引 5 是目标回复
                train_replies.append(terms[5].strip())
                
    print(f"✅ 成功提取到 {len(train_replies)} 条训练集候选回复。")

    # ================= 3. 提取测试集评论与真实回复 =================
    print(f"📖 正在读取测试集，提取原评论: {test_file}")
    test_reviews = []
    test_targets = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            terms = line.strip().split("-[split]-")
            if len(terms) >= 6:
                # 索引 4 是评论，索引 5 是回复
                test_reviews.append(terms[4].strip())
                test_targets.append(terms[5].strip())

    print(f"✅ 成功提取到 {len(test_reviews)} 条测试集数据。")

    # ================= 4. 随机抽取并按标准格式保存 =================
    print("🎲 正在为每条测试评论随机抽取回复...")
    
    # ⚠️ 学术规范提示：设定随机种子 (Random Seed)
    # 保证每次运行抽取的“随机”结果完全一致，确保你的实验具有绝对的可复现性。
    random.seed(42) 
    
    with open(out_pred_path, 'w', encoding='utf-8') as f_pred, \
         open(out_tar_path, 'w', encoding='utf-8') as f_tar:
        
        for rev, tar in zip(test_reviews, test_targets):
            # 从训练集回复池中随机挑一个
            random_reply = random.choice(train_replies)
            
            # 按 model.py 要求的格式写入文件
            f_pred.write(f"{rev}-[split]-{random_reply}\n")
            f_tar.write(f"{rev}-[split]-{tar}\n")

    # ================= 5. 完成提示 =================
    print("\n==================================================")
    print("✅ Random 基线实验数据生成完毕！")
    print(f"🎯 预测结果已保存至: {out_pred_path}")
    print(f"🎯 真实结果已保存至: {out_tar_path}")
    print("\n🚀 接下来，你可以直接用这行命令去计算 Random 基线的各项指标了：")
    print(f"python model.py --task evaluate --pred-file {out_pred_path} --no-test-simi")
    print("==================================================")

if __name__ == "__main__":
    run_random_baseline()

    