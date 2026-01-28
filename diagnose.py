import sys
import os
import inspect
import jax.numpy as jnp

# 强行把当前目录加到搜索路径，确保万无一失
sys.path.insert(0, os.getcwd())

print("="*60)
print(f"当前工作目录: {os.getcwd()}")
print("【第一步：验明正身】")

try:
    import JaxDFT.src.hamiltonian as ham
    print(f"✅ 成功导入模块: {ham.__file__}")
    
    # 获取内存中实际运行的代码
    src_lap = inspect.getsource(ham.laplacian_4th)
    src_pot = inspect.getsource(ham.gth_local_potential_value)
    
    print("-" * 30)
    # 检查动能算子系数
    if "3.0 *" in src_lap:
        print("✅ 拉普拉斯算子: 正常 (内存中检测到 3.0)")
    else:
        print("❌ 拉普拉斯算子: 错误! (没看到 3.0，代码没更新)")

    # 检查势能函数
    if "erf" in src_pot:
        print("✅ GTH势函数: 正常 (内存中检测到 erf)")
    else:
        print("❌ GTH势函数: 错误! (没看到 erf，代码没更新)")
        
except ImportError as e:
    print(f"❌ 依然无法导入 JaxDFT: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("【第二步：数值验证】")
try:
    # 构造 f = x^2 + y^2 + z^2，其拉普拉斯 Δf = 6.0
    spacing = 0.5
    N = 10
    x = jnp.linspace(-2.0, 2.0, N)
    X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')
    psi = X**2 + Y**2 + Z**2
    mask = jnp.ones_like(psi)
    
    # 计算动能项
    res = ham.laplacian_4th(psi, spacing, mask)
    center = N // 2
    val = float(res[center, center, center])
    
    print(f"动能算子测试 (理论值 6.0): {val:.5f}")
    if abs(val - 6.0) < 0.1:
        print("✅ 动能计算正确")
    else:
        print(f"❌ 动能计算错误 (你的结果是 {val:.5f}，说明系数不对)")

    # 计算势能项 (测试 r -> 0)
    r0 = 1e-12
    v0 = float(ham.gth_local_potential_value(r0, 1.0, 0.2, jnp.array([0.,0.,0.,0.])))
    print(f"势能原点测试 (r={r0}): {v0:.5f}")
    if abs(v0) > 10000:
        print("❌ 势能爆炸 (错误: 还在用 exp)")
    else:
        print("✅ 势能有限 (正确: 已使用 erf)")

except Exception as e:
    print(f"❌ 运行时报错: {e}")

print("="*60)
