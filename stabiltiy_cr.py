import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(page_title="压杆稳定 · 智能学习系统", layout="wide")

# ========== 侧边栏 ==========
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/column.png", width=80)
    st.title("📏 压杆稳定")
    st.markdown("**智能学习系统**")
    st.markdown("---")

    module = st.radio(
        "选择模块",
        ["📖 理论体系",
         "📊 临界力计算",
         "📈 长细比分析",
         "💪 稳定校核",
         "🏗️ 工程应用",
         "📐 案例分析",
         "🎨 应力云图",
         "🔄 3D失稳动画",
         "🧪 虚拟实验"]
    )
    st.markdown("---")
    st.caption("交互式学习 | 实时可视化 | 工程实战")

# ========== 材料数据库 ==========
MATERIALS = {
    "Q235钢": {"E": 210, "sigma_s": 235, "sigma_p": 200, "a": 304, "b": 1.12},
    "45钢": {"E": 210, "sigma_s": 355, "sigma_p": 300, "a": 450, "b": 1.47},
    "40Cr钢": {"E": 210, "sigma_s": 500, "sigma_p": 420, "a": 520, "b": 1.62},
    "铝合金": {"E": 70, "sigma_s": 280, "sigma_p": 240, "a": 310, "b": 1.15},
    "铸铁": {"E": 120, "sigma_s": 200, "sigma_p": 180, "a": 240, "b": 0.80},
}

def set_chinese_font(fig):
    fig.update_layout(
        font=dict(family="SimHei, Microsoft YaHei, Arial Unicode MS, sans-serif")
    )
    return fig

def euler_critical_load(E, I, mu, L):
    return np.pi**2 * E * I / (mu * L)**2

def slenderness_ratio(mu, L, i):
    return mu * L / i

# ============================================================
# 1. 理论体系
# ============================================================
if module == "📖 理论体系":
    st.title("📏 压杆稳定 · 理论体系")

    tab1, tab2, tab3, tab4 = st.tabs(["基本概念", "核心公式", "稳定分类", "知识图谱"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            ### 什么是压杆稳定？
            
            压杆稳定是指细长杆在轴向压力作用下，保持原有直线平衡状态的能力。
            
            ### 失稳现象
            当轴向压力超过某一临界值时，压杆突然发生侧向弯曲，这种现象称为失稳或屈曲。
            
            ### 工程实例
            - 建筑中的立柱
            - 起重机臂架
            - 桥梁桁架中的压杆
            - 机器人臂杆
            - 千斤顶螺杆
            """)
        with col2:
            st.markdown("""
            ### 失稳的特点
            
            - 突发性：无预兆，突然发生
            - 毁灭性：结构可能整体垮塌
            - 与强度破坏不同：失稳时应力可能远小于屈服强度
            
            ### 影响稳定性的因素
            1. 杆长：越长越不稳定
            2. 截面形状：惯性矩越大越稳定
            3. 约束条件：约束越强越稳定
            4. 材料性能：弹性模量越高越稳定
            """)

    with tab2:
        st.markdown("""
        ### 核心公式
        
        #### 1. 欧拉临界力公式
        $$P_{cr} = \\frac{\\pi^2 EI}{(\\mu L)^2}$$
        
        #### 2. 长度系数
        | 约束条件 | 失稳形状 | mu |
        |----------|----------|-----|
        | 两端铰支 | 正弦半波 | 1.0 |
        | 一端固定一端自由 | 四分之一正弦 | 2.0 |
        | 两端固定 | 全正弦波 | 0.5 |
        | 一端固定一端铰支 | 0.7l正弦 | 0.7 |
        
        #### 3. 长细比
        $$\\lambda = \\frac{\\mu L}{i}, \\quad i = \\sqrt{\\frac{I}{A}}$$
        
        #### 4. 稳定条件
        $$P \\leq \\frac{P_{cr}}{n_{st}}$$
        """)

    with tab3:
        st.markdown("""
        ### 压杆分类（按长细比）
        
        #### 1. 大柔度杆（细长杆）
        $$\\lambda \\geq \\lambda_p = \\pi \\sqrt{\\frac{E}{\\sigma_p}}$$
        
        #### 2. 中柔度杆（中长杆）
        $$\\lambda_s \\leq \\lambda < \\lambda_p$$
        
        #### 3. 小柔度杆（短粗杆）
        $$\\lambda < \\lambda_s$$
        """)

    with tab4:
        st.graphviz_chart('''
        digraph {
            "压杆稳定" -> "基本概念"
            "压杆稳定" -> "临界力"
            "压杆稳定" -> "长细比"
            "压杆稳定" -> "稳定校核"
            "临界力" -> "P_cr=pi^2EI/(mu L)^2"
            "临界力" -> "长度系数mu"
            "长细比" -> "lambda=mu L/i"
            "大柔度杆" -> "欧拉公式"
            "中柔度杆" -> "经验公式"
            "小柔度杆" -> "强度问题"
        }
        ''')

# ============================================================
# 2. 临界力计算
# ============================================================
elif module == "📊 临界力计算":
    st.title("📊 临界力计算器")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### 结构参数")
        L = st.slider("杆长 L (m)", 0.5, 6.0, 3.0, 0.1)
        end_cond = st.selectbox("约束条件", ["两端铰支", "一端固定一端自由", "两端固定", "一端固定一端铰支"])
        mu_map = {"两端铰支": 1.0, "一端固定一端自由": 2.0, "两端固定": 0.5, "一端固定一端铰支": 0.7}
        mu = mu_map[end_cond]

        st.markdown("### 截面参数")
        section_type = st.selectbox("截面形状", ["矩形", "圆形", "圆管"])
        if section_type == "矩形":
            b = st.slider("宽度 b (mm)", 20, 200, 50, 5)
            h = st.slider("高度 h (mm)", 20, 300, 80, 5)
            A = b * h
            I = b * h**3 / 12
        elif section_type == "圆形":
            d = st.slider("直径 d (mm)", 20, 200, 40, 2)
            A = np.pi * d**2 / 4
            I = np.pi * d**4 / 64
        else:
            D = st.slider("外径 D (mm)", 30, 250, 60, 5)
            t = st.slider("壁厚 t (mm)", 2, 30, 5, 1)
            A = np.pi * (D**2 - (D-2*t)**2) / 4
            I = np.pi * (D**4 - (D-2*t)**4) / 64

        material = st.selectbox("材料", list(MATERIALS.keys()))
        mat = MATERIALS[material]
        E = mat["E"] * 1e9

        A_m2 = A * 1e-6
        I_m4 = I * 1e-12
        i = np.sqrt(I / A) * 1e-3

        P_cr = euler_critical_load(E, I_m4, mu, L)
        lam = slenderness_ratio(mu, L, i)
        sigma_cr = P_cr / A_m2 / 1e6

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("欧拉临界力 P_cr", f"{P_cr/1000:.2f} kN")
            st.metric("临界应力 sigma_cr", f"{sigma_cr:.2f} MPa")
        with col_b:
            st.metric("长细比 lambda", f"{lam:.1f}")
            st.metric("惯性半径 i", f"{i*1000:.2f} mm")

    with col2:
        x = np.linspace(0, 1, 100)
        if end_cond == "两端铰支":
            y = np.sin(np.pi * x)
        elif end_cond == "一端固定一端自由":
            y = 1 - np.cos(np.pi * x / 2)
        elif end_cond == "两端固定":
            y = 1 - np.cos(2*np.pi * x)
        else:
            y = np.sin(np.pi * x)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x*L, y=y, mode='lines',
                                 name='失稳模态', line=dict(color='purple', width=4)))
        fig.add_annotation(x=L/2, y=0.8, text=f"P_cr = {P_cr/1000:.2f} kN", showarrow=False)
        fig.add_annotation(x=L/2, y=0.6, text=f"lambda = {lam:.1f}", showarrow=False)
        fig.update_layout(title=f"{end_cond} 一阶失稳模态", xaxis_title="杆长 (m)", yaxis_title="横向位移", height=350)
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 3. 长细比分析
# ============================================================
elif module == "📈 长细比分析":
    st.title("📈 长细比与稳定分类")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### 参数设置")
        L_range = st.slider("杆长范围 L (m)", 0.5, 10.0, (1.0, 6.0), 0.5)
        material = st.selectbox("材料", list(MATERIALS.keys()))
        mat = MATERIALS[material]
        E = mat["E"] * 1e9
        sigma_s = mat["sigma_s"]
        sigma_p = mat["sigma_p"]

        lam_p = np.pi * np.sqrt(E / (sigma_p * 1e6))

        b = st.slider("宽度 b (mm)", 20, 100, 50, 5)
        h = st.slider("高度 h (mm)", 20, 120, 80, 5)
        A = b * h
        I = b * h**3 / 12
        i = np.sqrt(I / A) * 1e-3

        end_cond = st.selectbox("约束条件", ["两端铰支", "一端固定一端自由", "两端固定"])
        mu_map = {"两端铰支": 1.0, "一端固定一端自由": 2.0, "两端固定": 0.5}
        mu = mu_map[end_cond]

    with col2:
        L_vals = np.linspace(L_range[0], L_range[1], 50)
        lam_vals = mu * L_vals / i
        sigma_cr_vals = np.zeros_like(lam_vals)

        for idx, lam in enumerate(lam_vals):
            if lam >= lam_p:
                sigma_cr_vals[idx] = np.pi**2 * E / lam**2 / 1e6
            else:
                sigma_cr_vals[idx] = mat["a"] - mat["b"] * lam

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=lam_vals, y=sigma_cr_vals, mode='lines',
                                 name='临界应力', line=dict(color='red', width=3)))
        fig.add_hline(y=sigma_s, line_dash="dash", line_color="green",
                     annotation_text=f"sigma_s={sigma_s}MPa")
        fig.add_vline(x=lam_p, line_dash="dash", line_color="blue",
                     annotation_text=f"lambda_p={lam_p:.1f}")
        fig.update_layout(title="长细比-临界应力曲线", xaxis_title="长细比 lambda", yaxis_title="临界应力 sigma_cr (MPa)", height=400)
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 4. 稳定校核
# ============================================================
elif module == "💪 稳定校核":
    st.title("💪 压杆稳定校核")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.markdown("### 设计参数")
        L = st.slider("杆长 L (m)", 0.5, 6.0, 3.0, 0.1)
        F = st.slider("工作压力 F (kN)", 10, 500, 100, 5)
        end_cond = st.selectbox("约束条件", ["两端铰支", "一端固定一端自由", "两端固定"])
        mu_map = {"两端铰支": 1.0, "一端固定一端自由": 2.0, "两端固定": 0.5}
        mu = mu_map[end_cond]

        st.markdown("### 截面参数")
        section_type = st.selectbox("截面形状", ["矩形", "圆形"])
        if section_type == "矩形":
            b = st.slider("宽度 b (mm)", 20, 200, 60, 5)
            h = st.slider("高度 h (mm)", 20, 300, 80, 5)
            A = b * h
            I = b * h**3 / 12
        else:
            d = st.slider("直径 d (mm)", 20, 200, 50, 2)
            A = np.pi * d**2 / 4
            I = np.pi * d**4 / 64

        material = st.selectbox("材料", list(MATERIALS.keys()))
        mat = MATERIALS[material]
        E = mat["E"] * 1e9
        n_st = st.slider("稳定安全系数 n_st", 1.5, 5.0, 2.5, 0.5)

        A_m2 = A * 1e-6
        I_m4 = I * 1e-12
        i = np.sqrt(I / A) * 1e-3

        P_cr = euler_critical_load(E, I_m4, mu, L)
        lam = slenderness_ratio(mu, L, i)
        P_allow = P_cr / n_st

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("临界力 P_cr", f"{P_cr/1000:.2f} kN")
            st.metric("许用载荷 P_allow", f"{P_allow/1000:.2f} kN")
        with col_b:
            st.metric("长细比 lambda", f"{lam:.1f}")
            st.metric("稳定安全系数", f"{P_cr/(F*1000):.1f}")

        if F <= P_allow/1000:
            st.success("稳定校核通过")
        else:
            st.error("稳定校核不通过")

    with col2:
        F_range = np.linspace(0, max(F*2, P_cr/1000*1.5), 50)
        P_cr_range = np.ones_like(F_range) * P_cr/1000

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=F_range, y=P_cr_range, mode='lines',
                                 name='临界力', line=dict(color='red', width=3)))
        fig.add_trace(go.Scatter(x=[0, max(F_range)], y=[P_allow/1000, P_allow/1000],
                                 mode='lines', name='许用载荷', line=dict(color='green', dash='dash')))
        fig.add_trace(go.Scatter(x=[F], y=[P_cr/1000], mode='markers',
                                 marker=dict(color='blue', size=16, symbol='star'),
                                 name='工作状态'))
        fig.update_layout(title="稳定安全区域", xaxis_title="工作载荷 F (kN)", yaxis_title="临界力 P_cr (kN)", height=400)
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 5. 工程应用
# ============================================================
elif module == "🏗️ 工程应用":
    st.title("🏗️ 工程应用案例")

    case = st.selectbox("选择案例", ["建筑立柱设计", "起重机臂架", "机器人臂杆稳定", "千斤顶螺杆"])

    if case == "建筑立柱设计":
        st.markdown("""
        ### 建筑钢立柱稳定设计
        
        **工程背景**：
        某建筑钢立柱承受轴向压力 F = 800 kN，柱高 L = 6 m，两端铰支。
        选用 Q235钢 工字形截面。要求设计最小截面尺寸。
        """)

        col1, col2 = st.columns(2)
        with col1:
            F = st.slider("压力 F (kN)", 200, 2000, 800, 50)
            L = st.slider("柱高 L (m)", 3, 12, 6, 0.5)
            material = st.selectbox("材料", ["Q235钢", "45钢", "40Cr钢"])
            n_st = st.slider("稳定安全系数", 2.0, 4.0, 2.5, 0.5)

        with col2:
            mat = MATERIALS[material]
            E = mat["E"] * 1e9

            b = st.slider("翼缘宽 b (mm)", 100, 400, 200, 10)
            h = st.slider("腹板高 h (mm)", 200, 600, 350, 10)
            tw = st.slider("腹板厚 tw (mm)", 6, 30, 12, 1)
            tf = st.slider("翼缘厚 tf (mm)", 6, 40, 16, 1)

            A = 2*b*tf + tw*(h-2*tf)
            I = (b*h**3 - (b-tw)*(h-2*tf)**3) / 12
            i = np.sqrt(I / A) * 1e-3
            A_m2 = A * 1e-6
            I_m4 = I * 1e-12

            mu = 1.0
            P_cr = euler_critical_load(E, I_m4, mu, L)
            lam = slenderness_ratio(mu, L, i)

            st.metric("临界力 P_cr", f"{P_cr/1000:.1f} kN")
            st.metric("长细比 lambda", f"{lam:.1f}")
            st.metric("截面积 A", f"{A:.0f} mm²")

            if P_cr/1000 >= F * n_st:
                st.success("稳定满足")
            else:
                st.error("不稳定，需增大截面")

# ============================================================
# 6. 案例分析
# ============================================================
elif module == "📐 案例分析":
    st.title("📐 案例分析：压杆稳定设计")

    st.markdown("""
    ### 问题描述
    
    一两端铰支的钢压杆，长度 L = 4 m，承受轴向压力 F = 200 kN。
    材料为 Q235钢。要求设计截面尺寸。
    """)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        F = st.slider("压力 F (kN)", 50, 500, 200, 10)
        L = st.slider("杆长 L (m)", 2, 8, 4, 0.5)
        material = st.selectbox("材料", list(MATERIALS.keys()))
        n_st = st.slider("稳定安全系数", 2.0, 4.0, 2.5, 0.5)

    with col2:
        mat = MATERIALS[material]
        E = mat["E"] * 1e9

        d_try = 50
        for i in range(20):
            A = np.pi * d_try**2 / 4
            I = np.pi * d_try**4 / 64
            i_r = np.sqrt(I / A) * 1e-3
            A_m2 = A * 1e-6
            I_m4 = I * 1e-12
            P_cr = euler_critical_load(E, I_m4, 1.0, L)
            if P_cr/1000 >= F * n_st:
                break
            d_try += 5

        h_try = 80
        for i in range(20):
            b_try = 0.5 * h_try
            A_rect = b_try * h_try
            I_rect = b_try * h_try**3 / 12
            i_rect = np.sqrt(I_rect / A_rect) * 1e-3
            A_m2_rect = A_rect * 1e-6
            I_m4_rect = I_rect * 1e-12
            P_cr_rect = euler_critical_load(E, I_m4_rect, 1.0, L)
            if P_cr_rect/1000 >= F * n_st:
                break
            h_try += 10

        st.metric("圆形直径", f"{d_try:.0f} mm")
        st.metric("矩形尺寸", f"{0.5*h_try:.0f} x {h_try:.0f} mm")

# ============================================================
# 7. 应力云图
# ============================================================
elif module == "🎨 应力云图":
    st.title("🎨 压杆应力云图")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        P = st.slider("压力 P (kN)", 0, 500, 100, 5)
        b = st.slider("截面宽 b (mm)", 20, 100, 50, 5)
        h = st.slider("截面高 h (mm)", 20, 150, 80, 5)
        L = st.slider("杆长 L (m)", 1.0, 5.0, 2.0, 0.5)
        view = st.selectbox("视图", ["压应力云图", "变形分布"])

    with col2:
        A = b * h / 1e6
        sigma = P * 1000 / A / 1e6

        if view == "压应力云图":
            y = np.linspace(-h/2, h/2, 30)
            z = np.linspace(-b/2, b/2, 30)
            Y, Z = np.meshgrid(y, z)
            stress_field = sigma * np.ones_like(Y)

            fig = go.Figure(data=go.Heatmap(
                z=stress_field, x=z, y=y,
                colorscale='Reds',
                colorbar=dict(title="sigma (MPa)")
            ))
            fig.update_layout(title=f"压应力分布 (sigma = {sigma:.2f} MPa)", height=450)
            fig = set_chinese_font(fig)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 8. 3D失稳动画
# ============================================================
elif module == "🔄 3D失稳动画":
    st.title("🔄 3D压杆失稳动画")

    col1, col2 = st.columns([1, 1.5])

    with col1:
        L = st.slider("杆长 L (m)", 1.0, 5.0, 3.0, 0.5)
        d = st.slider("直径 d (mm)", 20, 100, 40, 2)
        end_cond = st.selectbox("约束条件", ["两端铰支", "一端固定一端自由", "两端固定"])
        mu_map = {"两端铰支": 1.0, "一端固定一端自由": 2.0, "两端固定": 0.5}
        mu = mu_map[end_cond]
        load_ratio = st.slider("载荷比 P/P_cr", 0.0, 1.5, 0.8, 0.05)

    with col2:
        E = 210e9
        I = np.pi * d**4 / 64 * 1e-12
        P_cr = euler_critical_load(E, I, mu, L)

        st.metric("临界力 P_cr", f"{P_cr/1000:.2f} kN")

        x = np.linspace(0, L, 50)
        if end_cond == "两端铰支":
            y = np.sin(np.pi * x / L)
        elif end_cond == "一端固定一端自由":
            y = 1 - np.cos(np.pi * x / (2*L))
        else:
            y = 1 - np.cos(2*np.pi * x / L)

        if load_ratio <= 1.0:
            amp = 0
        else:
            amp = (load_ratio - 1.0) * 10

        y_def = y * amp

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=np.zeros_like(x), mode='lines',
                                 name='原始形状', line=dict(color='blue', dash='dot', width=3)))
        if load_ratio > 1.0:
            fig.add_trace(go.Scatter(x=x, y=y_def, mode='lines',
                                     name='失稳形状', line=dict(color='red', width=4)))
            fig.add_annotation(x=L/2, y=max(y_def)/2, text="失稳", showarrow=False, font_size=24)
        else:
            fig.add_annotation(x=L/2, y=0.5, text="稳定", showarrow=False, font_size=20)

        fig.update_layout(title=f"压杆稳定性 (P/P_cr = {load_ratio:.2f})",
                         xaxis_title="杆长 (m)", yaxis_title="横向位移", height=400)
        fig = set_chinese_font(fig)
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 9. 虚拟实验
# ============================================================
elif module == "🧪 虚拟实验":
    st.title("🧪 虚拟压杆实验")

    st.markdown("""
    ### 虚拟实验：压杆稳定
    
    逐步增加载荷，观察压杆失稳过程。
    """)

    col1, col2 = st.columns([1, 1.5])

    with col1:
        L = st.slider("杆长 L (mm)", 500, 3000, 1500, 100)
        b = st.slider("宽度 b (mm)", 10, 60, 30, 2)
        h = st.slider("高度 h (mm)", 10, 60, 40, 2)
        material = st.selectbox("材料", ["Q235钢", "铝合金", "钛合金"])
        E = {"Q235钢": 210, "铝合金": 70, "钛合金": 110}[material] * 1e9
        end_cond = st.selectbox("约束条件", ["两端铰支", "一端固定一端自由"])
        mu_map = {"两端铰支": 1.0, "一端固定一端自由": 2.0}
        mu = mu_map[end_cond]

        load_ratio = st.slider("加载比例", 0.0, 1.2, 0.0, 0.02)

    with col2:
        A = b * h * 1e-6
        I = b * h**3 / 12 * 1e-12
        L_m = L / 1000
        P_cr = euler_critical_load(E, I, mu, L_m)
        P = load_ratio * P_cr
        sigma = P / A / 1e6

        st.metric("临界力 P_cr", f"{P_cr/1000:.2f} kN")
        st.metric("当前载荷 P", f"{P/1000:.2f} kN")
        st.metric("当前应力", f"{sigma:.2f} MPa")

        if load_ratio < 1.0:
            st.success(f"稳定 (P/P_cr = {load_ratio:.2f} < 1.0)")
            st.progress(load_ratio)
        else:
            st.error(f"失稳！P/P_cr = {load_ratio:.2f} >= 1.0")
            st.balloons()

st.markdown("---")
st.caption("压杆稳定 · 智能学习系统 | 理论 + 计算 + 可视化")