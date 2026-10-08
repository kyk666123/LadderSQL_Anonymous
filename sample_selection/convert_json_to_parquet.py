import pandas as pd
import json

# 1. 读取 JSON 文件（假设文件名为 input.json）
input_json_path = "/path/to/dataset/SynSQL-2.5M/filtered_data_4000_ordered.json"
output_parquet_path = "/path/to/dataset/spider/train_only_synsql_4k.parquet"

with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 假设整个文件是一个 JSON 列表

# 2. 提取所需字段，并将 "sql" 重命名为 "query"
records = []
for item in data:
    records.append({
        "db_id": item["db_id"],
        "question": item["question"],
        "query": item["sql"]  # 注意：原字段是 "sql"
    })

# 3. 创建 DataFrame
df = pd.DataFrame(records)

# 4. 写入 Parquet 文件（使用 pyarrow 引擎）
df.to_parquet(output_parquet_path, engine="pyarrow", compression="snappy")

print(f"✅ 已成功将 {len(df)} 条记录写入 {output_parquet_path}")