import json
import re
from openai import OpenAI # 假设你用 OpenAI 兼容的接口调用 DeepSeek

# 初始化裁判客户端 (这里建议用你手里最强的模型作为裁判)
client = OpenAI(api_key="你的裁判API_KEY", base_url="你的裁判Base_URL")

def evaluate_single_response(user_query, reference_reply, model_generated_reply):
    """
    使用大模型作为裁判进行打分
    """
    
    # 核心：裁判的系统提示词
    system_prompt = """你是一个严谨客观的AI评测专家。你的任务是根据给定的【评分标准】，评估AI模型生成的回复质量。
    
【评分标准 (1-5分)】
1分：答非所问，或有严重事实错误。
2分：部分回答了问题，但遗漏了核心诉求，或语气生硬。
3分：基本回答了问题，无功无过，像模板回复。
4分：回答准确，语气自然友好，有同理心。
5分：完美解决诉求，语气极度专业且拟人，提供超预期的价值。

【输出要求】
请先深思熟虑地给出评价理由，最后给出具体的 1-5 分的整数。
强制要求：你必须且只能输出合法的 JSON 字符串，格式如下：
{
    "reasoning": "你的打分理由分析...",
    "score": 4
}
"""

    # 构造用户输入
    user_prompt = f"""
请评估以下数据：
【用户原始评论】: {user_query}
【官方参考回复】: {reference_reply}
【AI 生成的回复】: {model_generated_reply}

请给出你的 JSON 格式打分结果：
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # 或者 gpt-4o
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1, # 裁判必须冷静客观，温度设低一点
            response_format={"type": "json_object"} # 强制输出 JSON
        )
        
        result_text = response.choices[0].message.content
        
        # 解析 JSON
        result_json = json.loads(result_text)
        return result_json.get("score", 0), result_json.get("reasoning", "")
        
    except Exception as e:
        print(f"评测失败: {e}")
        return 0, "Error"

# ================= 测试一下 =================
if __name__ == "__main__":
    query = "这个App老是闪退，垃圾！赶紧退钱！"
    ref = "非常抱歉给您带来了不好的体验，请您提供一下设备型号，我们会尽快修复并为您处理退款事宜。"
    
    # 假设你的 CoRe 模型生成的回复
    core_reply = "亲爱的用户，实在是对不起！开发小哥已经在熬夜排查闪退问题了。关于退款，请您点击 App 右下角的客服通道，我会亲自为您加急办理。感谢您的包容！"
    
    score, reason = evaluate_single_response(query, ref, core_reply)
    print(f"打分: {score}/5")
    print(f"理由: {reason}")