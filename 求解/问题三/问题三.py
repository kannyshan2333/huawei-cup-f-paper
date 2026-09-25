# -*- coding: utf-8 -*-
"""
问题三：算力约束下的多维资源联合优化与结构性转移
输入：问题二广义标度律参数 L(N,D,Q)=E+A/N^a+B/D^b+C(1-Q)^g；
      问题一的质量基线 Q0 与配比修正 M(p)；附件 C7 的上下文长度可行取值。
成本（严格按赛题附录 B 与正文公式）：
      C_total = C_train + C_Q + C_attn
              = 6 N D + D * [g(Q) - g(Q0)]_+ + eta * N * D * L_ctx,  eta = 2e-4
      g(Q) 取附录 B 三型（指数型 1e7*exp(6Q)、幂函数型 5e9*Q^4、对数渐进型 2e9*ln(1+10Q)）
临界上下文长度：注意力开销 = 基础训练开销 <=> eta*N*D*L = 6*N*D <=> L_crit = 6/eta = 30000
方法：三档预算联合优化（SLSQP，对数参数化）；预算扫略 + 结构性转移识别；L_ctx 敏感性。
输出：图片到 图片/，结果 CSV 到 结果/。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import *   # noqa: F401,F403
from scipy.optimize import minimize

DATA = data_path

def save_csv_safe(df, name_cn):
    p = os.path.join(OUT_DIR, name_cn)
    try:
        df.to_csv(p, index=False, encoding='utf-8-sig')
    except PermissionError:
        alt = p.replace('.csv', '_new.csv')
        df.to_csv(alt, index=False, encoding='utf-8-sig')
        print('  ! %s 被占用，已改存为 %s' % (name_cn, os.path.basename(alt)))

# ============ 0. 读取前两问输出 ============
params = pd.read_csv(shared_path('广义标度律参数.csv'), index_col='param')['value']
E, A, alpha, B, beta, Cq, gamma = (params['E'], params['A'], params['alpha'],
                                   params['B'], params['beta'], params['C'], params['gamma'])

with open(shared_path('问题一_关键量.json'), encoding='utf-8') as fh:
    k1 = json.load(fh)
Q0_sample = float(k1['Q0_global_mean'])

mix_map = pd.read_csv(shared_path('17域质量映射.csv'))
ref_mix = pd.read_csv(data_path('train_mixture_1m.csv'))
ref_cols = [c for c in ref_mix if c.startswith('train_the_pile_')]
ref_p = ref_mix[ref_cols].mean().to_numpy(float)
ref_dom = [c.replace('train_the_pile_', '') for c in ref_cols]
qmap = mix_map.set_index('mixture_domain')['quality_Q']
q_vec = np.array([float(qmap.get(d, Q0_sample)) for d in ref_dom])
Q0 = float(ref_p @ q_vec)        # 基线质量 = 参考语料配比加权平均质量
print('=== 0. 质量基线与模型参数（来自问题一、二）===')
print('  参考语料配比加权基线质量 Q0 = %.4f（全量样本均值 %.4f）' % (Q0, Q0_sample))
print('  标度律：E=%.4f A=%.4f alpha=%.4f B=%.4f beta=%.4f C=%.4f gamma=%.4f'
      % (E, A, alpha, B, beta, Cq, gamma))

def loss(N_abs, D_abs, Q, M_p=0.0):
    """验证损失：参数以 B 为单位代入问题二标度律，M_p 为配比项修正。"""
    return (E + A * (N_abs / 1e9) ** (-alpha) + B * (D_abs / 1e9) ** (-beta)
            + Cq * (1 - Q) ** gamma + M_p)

# ============ 1. 附录 B 的三型质量成本函数 ============
GB = {
    '指数型': dict(gamma=1e7, lam=6.0),
    '幂函数型': dict(gamma=5e9, lam=4.0),
    '对数渐进型': dict(gamma=2e9, lam=10.0),
}
def g_of(Q, form):
    par = GB[form]
    if form == '指数型':
        return par['gamma'] * np.exp(par['lam'] * Q)
    if form == '幂函数型':
        return par['gamma'] * Q ** par['lam']
    return par['gamma'] * np.log(1 + par['lam'] * Q)

def c_quality(D_abs, Q, form):
    return D_abs * max(g_of(Q, form) - g_of(Q0, form), 0.0)

for form in GB:
    print('  g(%s) 在 Q0=%.3f 处为 %.4e，在 Q=1 处为 %.4e（单位算力/token）'
          % (form, Q0, g_of(Q0, form), g_of(1.0, form)))

# ============ 2. 临界上下文长度 ============
ETA = 2e-4
L_CRIT = 6.0 / ETA
arch = pd.read_csv(DATA('model_architecture_metadata.csv'))
L_FEAS = sorted(arch['max_position_embeddings'].dropna().unique().astype(int).tolist())
print('\n=== 1. 临界上下文长度（解析解）===')
print('  C_attn = eta*N*D*L_ctx, C_train = 6ND -> L_crit = 6/eta = %.0f tokens' % L_CRIT)
print('  C7 给出的可行上下文长度（max_position_embeddings）: %s' % L_FEAS)
print('  可行取值中跨越临界值的一对：8192（%.2f 倍）与 32768（%.2f 倍）'
      % (ETA * 8192 / 6, ETA * 32768 / 6))

# ============ 3. 总成本与单点最优 ============
def total_cost(N_abs, D_abs, Q, L_ctx, form):
    return 6.0 * N_abs * D_abs + c_quality(D_abs, Q, form) + ETA * N_abs * D_abs * L_ctx

def optimize(C_budget, L_ctx=2048, form='幂函数型', M_p=0.0, n_starts=8, Qcap=1.0):
    """SLSQP 对数参数化联合优化 (N, D, Q)。"""
    def obj(x):
        return loss(np.exp(x[0]), np.exp(x[1]), x[2], M_p)
    def con(x):
        return C_budget - total_cost(np.exp(x[0]), np.exp(x[1]), x[2], L_ctx, form)
    best = None
    for k in range(n_starts):
        frac = (k + 1) / (n_starts + 1)
        x0 = [np.log(1e9) + 4 * (frac - .5), np.log(1e11) + 4 * (.5 - frac),
              Q0 + (Qcap - Q0) * frac]
        try:
            res = minimize(obj, x0, method='SLSQP',
                           bounds=[(np.log(1e5), np.log(1e15)),
                                   (np.log(1e6), np.log(1e16)), (Q0, Qcap)],
                           constraints={'type': 'ineq', 'fun': con},
                           options={'maxiter': 1500, 'ftol': 1e-14})
        except Exception:
            continue
        if res.success and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError('优化失败')
    N_abs, D_abs, Q = np.exp(best.x[0]), np.exp(best.x[1]), best.x[2]
    c_base = 6.0 * N_abs * D_abs
    c_qual = c_quality(D_abs, Q, form)
    c_attn = ETA * N_abs * D_abs * L_ctx
    return dict(N_B=N_abs / 1e9, D_B=D_abs / 1e9, D_over_N=D_abs / N_abs, Q=Q,
                L=loss(N_abs, D_abs, Q, M_p), c_base=c_base, c_qual=c_qual,
                c_attn=c_attn, c_tot=c_base + c_qual + c_attn,
                f_base=c_base / C_budget, f_qual=c_qual / C_budget,
                f_attn=c_attn / C_budget, form=form, L_ctx=L_ctx, C=C_budget)

BUDGETS = [1e19, 1e22, 1e24]
print('\n=== 2. 三档预算最优配置（幂函数型质量成本，L_ctx=2048）===')
rows = []
for Cb in BUDGETS:
    r = optimize(Cb, 2048, '幂函数型')
    rows.append(r)
    print('  C=%.0e: N*=%.3fB D*=%.1fB Q*=%.4f L*=%.4f  '
          '[base %.1f%% qual %.1f%% attn %.1f%%]'
          % (Cb, r['N_B'], r['D_B'], r['Q'], r['L'],
             r['f_base'] * 100, r['f_qual'] * 100, r['f_attn'] * 100))
save_csv_safe(pd.DataFrame(rows), '三档预算最优配置.csv')

# ============ 4. 三种质量成本函数对比 ============
print('\n=== 3. 质量成本函数对最优解的影响 ===')
comp = []
for Cb in BUDGETS:
    for form in GB:
        r = optimize(Cb, 2048, form)
        comp.append(r)
comp_df = pd.DataFrame(comp)
save_csv_safe(comp_df, '成本函数对比.csv')
for Cb in BUDGETS:
    sub = comp_df[comp_df.C.eq(Cb)]
    print('  C=%.0e' % Cb)
    for _, r in sub.iterrows():
        print('    %-6s N*=%.3fB D*=%.1fB Q*=%.4f L*=%.4f 质量投入占比 %.1f%%'
              % (r['form'], r['N_B'], r['D_B'], r['Q'], r['L'], r['f_qual'] * 100))

# ============ 4b. 质量基线 Q0 口径的灵敏度 ============
_Q0_main = Q0
Q0 = Q0_sample
print('\n=== 3b. 质量基线口径灵敏度（Q0 由配比加权 %.4f 改为样本均值 %.4f）==='
      % (_Q0_main, Q0_sample))
q0_rows = []
for Cb in BUDGETS:
    r = optimize(Cb, 2048, '幂函数型')
    q0_rows.append(r)
    print('  C=%.0e: N*=%.3fB D*=%.1fB Q*=%.4f L*=%.4f 质量投入占比 %.1f%%'
          % (Cb, r['N_B'], r['D_B'], r['Q'], r['L'], r['f_qual'] * 100))
save_csv_safe(pd.DataFrame(q0_rows), '质量基线灵敏度.csv')
Q0 = _Q0_main

# ============ 5. 领域配比 M(p) 的算力等价效果 ============
mix_cal = pd.read_csv(os.path.join(SOLVE_DIR, '问题二', '结果', '配比项校准.csv'))
M_p = float(mix_cal.loc[mix_cal.mixture.eq('问题一有界推荐'), 'M_p'].iloc[0])
print('\n=== 4. 领域配比修正的算力等价效果（M(p)=%.4f）===' % M_p)
r_mix = optimize(1e22, 2048, '幂函数型', M_p=M_p)
r_ref = optimize(1e22, 2048, '幂函数型', M_p=0.0)
print('  C=1e22：参考配比 L*=%.4f（N*=%.3fB D*=%.1fB Q*=%.4f）'
      % (r_ref['L'], r_ref['N_B'], r_ref['D_B'], r_ref['Q']))
print('          推荐配比 L*=%.4f（N*=%.3fB D*=%.1fB Q*=%.4f）'
      % (r_mix['L'], r_mix['N_B'], r_mix['D_B'], r_mix['Q']))

def budget_for_loss(L_target, L_ctx=2048, form='幂函数型'):
    lo, hi = 1e17, 1e25
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        try:
            r = optimize(mid, L_ctx, form, n_starts=3)
            if r['L'] > L_target:
                lo = mid
            else:
                hi = mid
        except Exception:
            lo = mid
    return np.sqrt(lo * hi)

C_eq = budget_for_loss(r_mix['L'])
print('  推荐配比在 C=1e22 时得到的损失 %.4f，若改用参考配比实现同一损失需要'
      ' C≈%.3e FLOPs，即配比优化的效果等价于约 %.1f 倍算力扩张'
      % (r_mix['L'], C_eq, C_eq / 1e22))
print('  （说明：M(p) 是损失水平上的固定平移，与算力无关，故换算为算力收益会被'
      '损失的算力弹性放大，此倍数宜作量级参考。）')
save_csv_safe(pd.DataFrame([{'mixture': '参考配比', 'M_p': 0.0, **r_ref},
                            {'mixture': '问题一有界推荐', 'M_p': M_p, **r_mix}]),
              '配比项算力等价.csv')

# ============ 6. 预算扫略与结构性转移识别 ============
print('\n=== 5. 预算扫略与结构性转移 ===')
C_sweep = np.logspace(18, 25, 141)
sweep = []
for Cb in C_sweep:
    try:
        sweep.append(optimize(Cb, 2048, '幂函数型', n_starts=5))
    except Exception:
        sweep.append(dict(N_B=np.nan, D_B=np.nan, Q=np.nan, L=np.nan, c_base=np.nan,
                          c_qual=np.nan, c_attn=np.nan, c_tot=np.nan, f_base=np.nan,
                          f_qual=np.nan, f_attn=np.nan, form='幂函数型', L_ctx=2048, C=Cb))
sw = pd.DataFrame(sweep)
save_csv_safe(sw, '预算扫略.csv')

Qc = sw['Q'].to_numpy(float)
logC = np.log10(C_sweep)
ok = np.isfinite(Qc)
# 结构性转移的定量定义：
#   以“质量投入份额 f_qual 的相对变化方向反转 + 最优质量对预算的弹性改变符号/量级”为标志。
inQ = (Qc - Q0) / (1 - Q0)                     # 质量提升完成度
d_inQ = np.gradient(inQ[ok], logC[ok])
fQ = sw['f_qual'].to_numpy(float)
d_fQ = np.gradient(fQ[ok], logC[ok])
shift = {'质量提升完成度弹性 dln(Theta)/dlnC 的最大值点预算': float(C_sweep[ok][np.argmax(d_inQ)]),
         '质量投入启动边界（份额由零转正的拐点预算）': float(C_sweep[ok][np.argmax(d_fQ)]),
         '质量投入份额峰值预算': float(C_sweep[ok][np.argmax(fQ[ok])]),
         '质量投入份额峰值': float(np.nanmax(fQ[ok]))}
# 以"质量提升完成度" Theta=(Q*-Q0)/(1-Q0) 报告门槛，避免 Q0 本身已高于固定阈值时失去意义
for frac in [0.25, 0.5, 0.75, 0.99]:
    hit = Qc[ok] >= Q0 + frac * (1 - Q0)
    shift['完成度Theta首次达到%d%%的预算' % int(frac * 100)] = (
        float(C_sweep[ok][np.argmax(hit)]) if hit.any() else float('nan'))
for k, v in shift.items():
    print('  %s = %.3e FLOPs' % (k, v))

# N/D 比随预算的演化（规模结构）
sw['D_over_N'] = sw['D_B'] / sw['N_B']
print('  D/N 比：C=1e19 时 %.1f，C=1e22 时 %.1f，C=1e24 时 %.1f'
      % tuple(sw.loc[(sw.C - c).abs().idxmin(), 'D_over_N'] for c in [1e19, 1e22, 1e24]))
save_csv_safe(pd.DataFrame([shift]), '结构性转移识别.csv')

# ============ 7. 上下文长度敏感性（C7 可行取值 + 临界值） ============
print('\n=== 6. 上下文长度敏感性（C=1e22，幂函数型）===')
sens = []
for L in L_FEAS:
    r = optimize(1e22, L, '幂函数型')
    r['attn_over_train'] = ETA * L / 6.0
    r['is_above_crit'] = L > L_CRIT
    sens.append(r)
    print('  L=%6d（%.2f 倍训练开销）: N*=%.3fB D*=%.1fB Q*=%.4f L*=%.4f 注意力占比 %.1f%%%s'
          % (L, r['attn_over_train'], r['N_B'], r['D_B'], r['Q'], r['L'],
             r['f_attn'] * 100, '  [超临界]' if L > L_CRIT else ''))
sens_df = pd.DataFrame(sens)
save_csv_safe(sens_df, '上下文长度敏感性.csv')

# 连续 L 扫描（含临界点）
L_grid = np.logspace(np.log10(1024), np.log10(200000), 40)
cont = []
for L in L_grid:
    try:
        r = optimize(1e22, float(L), '幂函数型', n_starts=3)
        r['attn_over_train'] = ETA * L / 6.0
        cont.append(r)
    except Exception:
        pass
save_csv_safe(pd.DataFrame(cont), '上下文长度连续扫描.csv')

# ============ 8. 绘图 ============
plt.rcParams['font.size'] = 12
C1_, C2_, C3_ = '#3b6ea5', '#e07a3f', '#5aa05a'

opt_df = pd.DataFrame(rows)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
x = np.arange(len(BUDGETS))
axes[0].plot(x, opt_df['N_B'], 'o-', lw=2, ms=8, color=C1_, label='参数量 N* (B)')
axes[0].plot(x, opt_df['D_B'], 's--', lw=2, ms=8, color=C2_, label='数据量 D* (B tokens)')
for xi, (nb, db) in enumerate(zip(opt_df['N_B'], opt_df['D_B'])):
    axes[0].annotate('%.3g' % nb, (xi, nb), fontsize=8, xytext=(0, 8),
                     textcoords='offset points', ha='center', color=C1_)
    axes[0].annotate('%.4g' % db, (xi, db), fontsize=8, xytext=(0, -15),
                     textcoords='offset points', ha='center', color=C2_)
axes[0].set_yscale('log'); axes[0].set_xticks(x)
axes[0].set_xticklabels(['10¹⁹', '10²²', '10²⁴'])
axes[0].set_xlabel('算力预算 C (FLOPs)'); axes[0].set_ylabel('最优规模（对数尺度）')
axes[0].set_title('(a) 最优参数量与数据量'); axes[0].legend(fontsize=9); despine(axes[0])
axes[1].plot(x, opt_df['Q'], 'o-', lw=2, ms=8, color=C3_, label='最优质量 Q*')
axes[1].axhline(Q0, color='gray', ls=':', lw=1.2, label='基线质量 Q₀=%.3f' % Q0)
for xi, qv in enumerate(opt_df['Q']):
    axes[1].annotate('%.3f' % qv, (xi, qv), fontsize=9, xytext=(0, 8),
                     textcoords='offset points', ha='center', color=C3_)
axes[1].set_xticks(x); axes[1].set_xticklabels(['10¹⁹', '10²²', '10²⁴'])
axes[1].set_ylim(0, 1.12); axes[1].set_xlabel('算力预算 C (FLOPs)')
axes[1].set_ylabel('最优质量 Q*'); axes[1].set_title('(b) 最优质量随预算的变化')
axes[1].legend(fontsize=9); despine(axes[1])
fig.tight_layout(); save_fig(fig, '图1_三档预算最优配置')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
axes[0].plot(C_sweep, Qc, lw=2, color=C3_)
axes[0].axhline(Q0, color='gray', ls=':', lw=1.2, label='基线质量 Q₀=%.3f' % Q0)
axes[0].axvline(shift['质量投入启动边界（份额由零转正的拐点预算）'], color='#d62728', ls='--', lw=1.2,
                label='投入份额拐点')
axes[0].set_xscale('log'); axes[0].set_xlabel('算力预算 C (FLOPs)')
axes[0].set_ylabel('最优质量 Q*'); axes[0].set_ylim(0, 1.05)
axes[0].set_title('(a) 最优质量随预算的演化'); axes[0].legend(fontsize=8); despine(axes[0])
axes[1].plot(C_sweep, sw['f_base'] * 100, lw=2, color=C1_, label='基础训练 6ND')
axes[1].plot(C_sweep, sw['f_qual'] * 100, lw=2, color=C2_, label='质量提升 D[g(Q)-g(Q₀)]')
axes[1].plot(C_sweep, sw['f_attn'] * 100, lw=2, color='#d62728', label='长文本注意力')
axes[1].set_xscale('log'); axes[1].set_xlabel('算力预算 C (FLOPs)')
axes[1].set_ylabel('预算份额 (%)'); axes[1].set_title('(b) 预算份额的结构性演化')
axes[1].legend(fontsize=8); despine(axes[1])
fig.tight_layout(); save_fig(fig, '图2_预算份额结构性转移')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
axes[0].plot([r['attn_over_train'] for r in cont], [r['N_B'] for r in cont], 'o-', ms=3, color=C1_)
axes[0].set_xscale('log'); axes[0].set_yscale('log')
axes[0].axvline(1.0, color='#d62728', ls='--', lw=1.2,
                label='临界 L*=%.0f（注意力=训练）' % L_CRIT)
axes[0].set_xlabel('注意力开销 / 训练开销 = ηL/6'); axes[0].set_ylabel('N* (B)')
axes[0].set_title('(a) 上下文长度对最优规模的影响'); axes[0].legend(fontsize=8); despine(axes[0])
# 右图：不同预算档下 Q* 随注意力开销倍数的变化（低预算档才有内点解，故三档并列）
L_small = np.logspace(np.log10(1024), np.log10(150000), 14)
_qL_rows = []
for Cb, col, lbl in [(1e19, C3_, '10¹⁹'), (1e20, C2_, '10²⁰'), (1e21, C1_, '10²¹')]:
    xs, ys = [], []
    for L in L_small:
        try:
            rr = optimize(Cb, float(L), '幂函数型', n_starts=3)
            xs.append(ETA * L / 6.0); ys.append(rr['Q'])
            _qL_rows.append({'C': Cb, 'L_ctx': float(L), 'attn_over_train': ETA * L / 6.0,
                             'N_B': rr['N_B'], 'D_B': rr['D_B'], 'Q': rr['Q'], 'L': rr['L']})
        except Exception:
            pass
    axes[1].plot(xs, ys, 'o-', ms=3, color=col, label='预算 %s FLOPs' % lbl)
save_csv_safe(pd.DataFrame(_qL_rows), '上下文长度与最优质量.csv')
axes[1].axvline(1.0, color='#d62728', ls='--', lw=1.2, label='临界 L*')
axes[1].set_xscale('log'); axes[1].set_xlabel('注意力开销 / 训练开销 = ηL/6')
axes[1].set_ylabel('Q*'); axes[1].set_ylim(0, 1.05)
axes[1].yaxis.get_major_formatter().set_useOffset(False)
axes[1].set_title('(b) 上下文长度对质量投入的影响')
axes[1].legend(fontsize=8); despine(axes[1])
fig.tight_layout(); save_fig(fig, '图3_上下文长度敏感性')

fig, ax = plt.subplots(figsize=(7, 5))
Q_g = np.linspace(Q0, 1.0, 300)
for form, lbl in [('幂函数型', '幂函数型 5×10⁹Q⁴'),
                  ('指数型', '指数型 10⁷e^(6Q)'),
                  ('对数渐进型', '对数渐进型 2×10⁹ln(1+10Q)')]:
    ax.plot(Q_g, [g_of(q, form) - g_of(Q0, form) for q in Q_g], lw=2, label=lbl)
ax.set_xlabel('数据质量 Q'); ax.set_ylabel('单位 token 质量成本 g(Q)-g(Q₀)')
ax.set_title('附录 B 三种质量成本函数（Q₀=%.3f）' % Q0)
ax.legend(fontsize=9); despine(ax); fig.tight_layout(); save_fig(fig, '图4_质量成本函数对比')

# ============ 9. 汇总 ============
print('\n===== 问题三求解完成 =====')
print('  临界上下文长度 L_crit = 6/eta = %.0f tokens；C7 可行取值 %s' % (L_CRIT, L_FEAS))
print('  三档预算（幂函数型）：' +
      '；'.join('C=%.0e -> N*=%.3fB D*=%.1fB Q*=%.4f L*=%.4f'
                % (r['C'], r['N_B'], r['D_B'], r['Q'], r['L']) for r in rows))
