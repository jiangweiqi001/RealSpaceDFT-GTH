import os

path = 'scripts/verify_h2.py'
with open(path, 'r', encoding='utf-8') as f:
    code = f.read()

# 替换参数设置部分，引入自适应步长和500次迭代
old_params = """spacing = 0.18
box_size = [16.0, 16.0, 16.0]"""

new_params = """target_spacing = 0.18
L = 18.0  # 对齐 check_box.py 的 Target
N = int(round(L / target_spacing))
spacing = L / N
box_size = [L, L, L]
max_iter = 500"""

code = code.replace(old_params, new_params)

# 替换 solver.energy_and_forces 里的硬编码 100
code = code.replace("100, 0.3, 1e-5, key", "max_iter, 0.3, 1e-5, key")

with open(path, "w", encoding="utf-8") as f:
    f.write(code)

print("✅ scripts/verify_h2.py 参数与自适应网格修正完成！")
