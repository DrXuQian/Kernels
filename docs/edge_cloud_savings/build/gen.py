# -*- coding: utf-8 -*-
"""Generate the rewritten DingTalk-style docx from B's package + new content."""
import os, re, shutil, zipfile, struct, sys
from xml.sax.saxutils import escape as _esc

SRC_DOCX = sys.argv[1]          # original B.docx
OUT_DOCX = sys.argv[2]          # output path
HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "1E6FD9"
GREY = "8A8F99"
TEXT_W = 11220                  # twips: 13380 - 2*1080
EMU_PER_TWIP = 635

# ----------------------------------------------------------------------------
# content
# ----------------------------------------------------------------------------
DOC = []
def add(kind, *args, **kw): DOC.append((kind, args, kw))

add('title', '边缘AI·惊喜开箱4：端云协同省钱计划')
add('tag', '别让每个 Token 都上云。端侧 AI Box 不只是一台跑模型的盒子，更是给你的 Agent 配的一位【本地店长】：日常的活儿自己干，真遇到疑难杂症，再按铃叫云端专家。账单变薄，数据不出门。')
add('tag', '花十分钟读完，你会🉐到三件事：第一章，端云协同省的到底是哪本账；第二章，照着 Perplexity 和 NVIDIA Switchyard 抄作业，把「本地当店长、云端做会诊」的架构搭起来；第三章，编码、客服、办公助理各能省多少，以及怎么落地、怎么验收。')

TOC = [
    ('一、先把账算明白：Token 没消失，只是换了收银台', ['1.1 三本账，别混着算', '1.2 比较基线：每一次调用都走 Opus', '1.3 27B、0.6B 只是示意，不是采购单']),
    ('二、架构：本地当店长，云端做会诊', ['2.1 抄谁的作业：Perplexity 管「家」，Switchyard 管「路」', '2.2 五步闭环，一图看完', '2.3 谁干什么、不许干什么', '2.4 隐私不靠提示词，靠两道会关上的门']),
    ('三、能省多少，怎么落地', ['3.1 三次截流：钱是这么省下来的', '3.2 分场景数据：每个数字都带身份证', '3.3 换算成人民币', '3.4 落地三步走：先看清，再设门，最后调优', '3.5 验收看板：别只盯「省了百分之几」', '3.6 边界与风险：不是免费午餐，也不是万能防护罩']),
    ('结论：让云端更贵重，而不是更常驻', []),
    ('附：数据来源', []),
]
add('toc', TOC)

# ---------------- 一 ----------------
add('h2', '一、先把账算明白：Token 没消失，只是换了收银台')
add('p', '把 AI 系统想成一家医院。纯云方案是：挂号、量体温、开个感冒药、做疑难会诊，通通找顶级专家。效果不差，但专家的号很贵，而且你的病历得一次次送出门。端云协同干的事很朴素：让本地团队先分诊、先处理常规项目、先把材料整理好；只有本地确实扛不住的那一步，才把**最小必要信息**递给云端专家。')

add('h3', '1.1 三本账，别混着算')
add('p', '一聊「省 Token」，很多人脑子里只有一本账。其实至少有三本：')
add('table', dict(cols=[2600, 3800, 4820], header=['口径', '它在回答什么', '端云协同之后'], rows=[
    ['**总推理 Token**', '本地 + 云端一共嚼了多少内容', '不一定变少。本地做规划、分类、汇总，同样在吐 Token'],
    ['**云端出站 Token**', '有多少上下文真的离开了设备', '变少。本地先干、上下文裁剪、最小暴露，三招都在压它'],
    ['**云端计费 Token**', '有多少 Token 进了云端的收银台', '变少。云端调用比例降 + 每次送的内容少，账单直接瘦身'],
]))
add('p', '所以准确的说法不是「端云协同让 Token 凭空消失」，而是：**把原本默认送云的活儿留在本地，把必须升级的那部分精简后再送云。** Token 没少，只是换了一家收银台——本地这家收银台不按 Token 收钱，只收电费。')

add('h3', '1.2 比较基线：每一次调用都走 Opus')
add('p', '本文所有「省 X%」都拿同一把尺子量：**每一次调用都走 Opus**。本地模型只算电费和硬件，不进云端账单；「云端剩余」指路由之后仍然要云端处理的比例。')
add('p', '这把尺子擅长回答「云账单能少多少」，但回答不了「系统总成本一定少多少」——盒子的硬件和电费在另一张账上。这一点先说清楚，后面的数字才不会被读歪。')

add('h3', '1.3 27B、0.6B 只是示意，不是采购单')
add('p', '文中出现的本地 27B 主模型、0.6B PII 模型、256K 上下文、vLLM，以及 Qwen3.8-Flash、DeepSeek V4 Flash 这些名字，都是**架构示意或候选替换**，不是必须绑定的采购清单。27B 可以换成更大的单模型，也可以换成几个小模型搭班子。你要决策的是「能力怎么分工」，不是「买哪个型号」。把架构绑死在一个模型名字上，等于把「能力可替换」做成了「型号依赖」——第三章的风险清单会专门点名这个坑。')
add('hr')

# ---------------- 二 ----------------
add('h2', '二、架构：本地当店长，云端做会诊')
add('img', 'meme_all.png', 2.6)
add('p', '「省钱」和「隐私」这两件事，很多方案只能二选一：要么全上云、图省事；要么全本地、图安全。端云协同的态度是——我全都要。做法也不神秘，业界已经有两份现成的作业可以抄。')

add('h3', '2.1 抄谁的作业：Perplexity 管「家」，Switchyard 管「路」')
add('p', '**Perplexity Portable Computer，负责「家里怎么布置」。** 2026 年 2 月 Perplexity 发布了 Computer：一个会拆任务、派子代理、在沙箱里跑工具、还接 Slack / GitHub / 邮件的多模型 Agent。8 月它又和 NVIDIA 一起推出了完全本地版 Portable Computer——编排模型、子代理模型、Agent 运行时、工具沙箱、连接器，全部跑在你自己的硬件上（DGX Spark，或任何 24GB 显存以上的 RTX 卡），本地模型是一个专门为 Agent 后训练过的 27B。本地完成的步骤**不消耗一分钱云端额度**；真需要前沿推理时，编排器会先弹窗要你批准，出站内容先过 PII 标记，云端只能回文字建议。沙箱是操作系统级强制隔离，沙箱不可用就直接禁用工具执行，绝不「悄悄降级」。本章后面的五步闭环、双闸门、fail-closed，基本就是这套思路。')
add('p', '**NVIDIA NeMo Switchyard，负责「什么时候出门」。** 它是 NVIDIA 开源的模型路由层，兼容 OpenAI 与 Anthropic 接口，专门回答一个问题：Agent 的这一次调用，到底要不要花前沿模型的钱？它自带几种现成的路由器：')
add('bullets', [
    '**Escalation（升级路由）**：先用便宜模型干，裁判模型盯着看；连续两次判定「卡住了」，这个任务之后就全交给贵模型。单向升级，不反悔。',
    '**Stage（阶段路由）**：看最近的工具结果和进度。稳定改代码、测试已过，就用便宜的；报错连连、原地打转，就换贵的。',
    '**Capability / LLM 分类**：先让裁判判断任务难度，选定模型后在后续轮次保持会话亲和，不反复分类。',
    '**Prefill（可训练路由）**：从模型残差流里读信号，预测每个候选模型的成功概率，再按成本、延迟做取舍。',
])
add('p', '第三章那些「云端剩余 X%」，大多就是这些路由器在 Terminal-Bench 2.1、τ²-bench 和 LangChain 任务集上跑出来的。一句话：**Perplexity 告诉你本地这个家怎么布置，Switchyard 告诉你什么时候该出门、出门带多少东西。**')

add('h3', '2.2 五步闭环，一图看完')
add('img', 'meme_niubi.png', 2.6)
add('p', '端云协同不是在请求前面加一个分类器就完事，而是一条「本地默认、升级受控、结果回流」的闭环。先上图：')
add('img', 'fig_arch.png', 'full')
add('p', '图里三种颜色对应三块地盘：GPU 区常驻两个模型（27B 主推理 + 0.6B PII 审查），CPU 区跑 Agent 运行时、子代理和两套沙箱，右边橙色是受控访问的云端。整台设备只留**一个网络出口**，而且只有 Agent 内置的出网门控能走。')
add('p', '**第一步：用户下达任务。** 请求先进本地编排器。会话、本地文件、任务上下文默认留在设备上，不会把整段对话自动打包上云。')
add('p', '**第二步：本地模型先上。** 27B 主模型负责规划、能力分类和编排决策；能本地完成的日常任务直接完成，需要工具就交给本地沙箱执行——沙箱隔离运行环境，零出网。')
add('p', '**第三步：确需云端，先过隐私闸门。** 本地判断这一步需要前沿推理或实时数据，不是立刻联网，而是先裁剪上下文，只留任务需要的最小片段；再送 0.6B PII 模型审查；然后把「即将离开设备的确切内容」摆到你面前，等你点头。')
add('p', '**第四步：云端专家按需处理。** 你批准了，云端模型群才上场。云端碰不到本地文件，调不了本地工具，看不到完整对话，也不能主动往回推东西；它只收到批准过的片段，只能回一段文字建议。')
add('p', '**第五步：结果回本地继续跑。** 云端输出不是直接接管系统，而是回到本地编排器，由本地模型结合本地状态、工具和后续步骤继续执行。')
add('p', '所以云端是「按铃才来的会诊专家」，不是一直坐在驾驶座上的默认司机。')

add('h3', '2.3 谁干什么、不许干什么')
add('table', dict(cols=[2700, 4400, 4120], header=['组件', '主要职责', '明确边界'], rows=[
    ['本地编排器 / 主模型', '规划、分类、路由、常规推理、上下文整理', '规格可替换，不绑定 27B'],
    ['本地工具沙箱', '执行工具与任务动作', '隔离环境、零出网'],
    ['0.6B PII 审查模型', '对出站内容做实体级检测', '常驻本地流水线，规格为材料示意'],
    ['本地隔离子代理', '处理涉密计算，只回传非身份汇总', '沙箱不可用则禁用，不降级裸奔'],
    ['云端模型群', '本地啃不动的前沿推理、实时数据类任务', '无本地工具权、文件权和完整会话权'],
]))

add('h3', '2.4 隐私不靠提示词，靠两道会关上的门')
add('p', '只在系统提示里写一句「请注意隐私」，就像在门口贴张纸条「贵重物品请勿带出」——有礼貌，没用。端云协同要的是两道真的会关上的门。')
add('p', '**闸门一：出站前审查 + 逐次批准。** 所有待出站内容先过本地 PII 模型：分词后进 0.6B 双向编码器，对 37 类实体逐 Token 打分，再用约束 Viterbi 解码出实体边界，最后决定脱敏还是拦截。给用户看的时候归成 9 类：人名、邮箱、电话、地址、URL、日期、账号、机密事项和其他。')
add('p', '审查不是「授权一次，永久放行」。系统每次都把将要出门的确切内容摆出来，你批准这一次、这一小段，才发。批准的是一次出行，不是给整台设备发通行证。')
add('p', '材料里的 PII 指标：4096 Token 上下文 + 50% 重叠滑窗，长文本 recall 从 0.83 拉到 0.97；字符级 F1 0.629，材料称在 12 个评测系统中最高；重复 PII 一致检出率 79.4%，对比前沿云端模型 57.0%；BF16 部署，与主模型共享 GPU，显存预算只占 5%。这些是材料中的设计与评测记录，落地时仍要在自己的数据和链路上复验。')
add('p', '**闸门二：涉密计算进本地隔离子代理。** 碰敏感数据的计算，不为了「方便」把原始内容升级上云，而是派给独立沙箱里的本地子代理。子代理只回传非身份汇总，涉密数据留在隔离区。')
add('p', '最关键的一条是 **fail-closed**：沙箱不可用，功能直接禁用，不做无保护降级。很多系统的坏习惯是「安全组件挂了，先把任务跑完再说」；这里反过来——门锁坏了，门就一直关着。')
add('p', '两道门之外，还有云端最小权限：无本地文件权、无本地工具权、无完整会话权、不能主动推送，只能对批准过的片段回文字建议。就算叫了云端，暴露面也只有这一次任务的最小集合。')
add('hr')

# ---------------- 三 ----------------
add('h2', '三、能省多少，怎么落地')
add('img', 'meme_money.png', 2.4)

add('h3', '3.1 三次截流：钱是这么省下来的')
add('p', '省钱不是一个开关，而是三次「截流」：')
add('p', '**第一次截流：本地直接干完。** 一轮任务不需要升级，云端剩余就少一轮，云端计费也少一轮。')
add('p', '**第二次截流：只升级难点。** 多轮任务不必从第一句到最后一句全绑在云端，可以按轮、按能力、按阶段路由。云端从「全程包办」变成「疑难会诊」。')
add('p', '**第三次截流：只送最小必要上下文。** 即便升级，也不默认把本地文件、完整对话、全部工具状态一股脑发出去，而是本地裁剪、汇总后再出站。「调用次数降」和「单次出站内容降」两个杠杆同时压在云端计费 Token 上。')
add('p', '真正决定收益的，不是「有没有本地模型」，而是本地模型能独立扛住多少轮、多少阶段，以及升级时暴露多少上下文。这也是下面各场景差异这么大的原因。')

add('h3', '3.2 分场景数据：每个数字都带身份证')
add('img', 'fig_savings.png', 'full')
add('quote', '下表的「节省」都是相对「每次调用都走 Opus」的云端账单节省，不等于总推理 Token 减少，也没算本地硬件与电费。')
add('table', dict(cols=[2100, 1800, 1150, 1400, 2200, 2570], size=18,
    header=['场景', '路由方式', '云端剩余', '云端账单节省', '质量 / 准确率', '数据属性与限制'], rows=[
    ['编码·Terminal-Bench 2.1', 'Escalation，两次升级锁定', '51–61%', '**40–50%**', '75.7% vs 76.0%，-0.3', 'NVIDIA 官方 + 推算拆账；可复现'],
    ['编码·Stage 无裁判', 'Stage', '55–59%', '41–45%', '72.7%，-3.3', '官方 + 推算；参数在 README'],
    ['编码·Ramp SWE-Bench', 'Stage', '32–38%', '62–68%', '与前沿模型持平', '第三方 + 推算；任务集内部'],
    ['编码·FrontierCode Main', 'Stage', '50–61%', '40–50%', '-2.8 分', '第三方 + 推算；任务私有'],
    ['编码·opencode', 'Capability 分类', '50–55%', '45–50%', '13 任务全对', '第三方 + 推算；配置与命令公开'],
    ['编码·全本地', '全部本地', '0%', '100%', '未在同一 harness 验证', '推算；可自测，只宜作路线图'],
    ['客服多轮·平衡档，τ²-bench', 'Custom 分类，每用户轮', '55–65%', '35–45%', '0.903 ± 0.071', '官方轮次占比 + Token 推算；可复现'],
    ['客服多轮·激进档，τ²-bench', '同上', '15–25%', '**75–85%**', '0.891 ± 0.029', '官方轮次占比 + Token 推算；可复现'],
    ['多轮 Agent', 'Escalation', '约 18%', '约 82%', '-6 分', '第三方 + 推算；参数公开、评测集内部'],
    ['个人助理型', 'Escalation + 按类型分流（推测）', '约 27%', '**70–75%**', '79.4% vs 79.3%', '官方总体结果 + 推算；配置未公开'],
    ['企业域路由', '域分类', '约 41%（按流量）', '55–60%', '域路由准确率 100%，后续轮延迟 -21%', '第三方 + 推算；不可复现'],
    ['视频生成', '全本地', '0%', '100%', '未给出', '本方实测；可复现'],
]))
add('p', '适合写进决策摘要的保守读法：编码 / 运维场景盯 ==40–50%==（官方或第三方结果 + 推算，具体看上表）；客服多轮激进档 ==75–85%==（官方轮次占比 + Token 推算）；办公助理与工作流自动化 ==70–82%==（官方或第三方结果 + 推算，部分配置或评测集未公开）。这些是可能区间，不是跨场景承诺。')
add('p', '顺手解释一个常被问到的差异：Switchyard 官方账本里，本地那个 Nemotron 也是按 API 单价计费的，所以官方给的总成本降幅（比如 Terminal-Bench 2.1 上升级路由只便宜 13.3%）比本文的「云端账单节省」低得多。本文的口径是本地模型跑在自己的盒子上、只交电费，只数云端那部分。这是口径差，不是魔法——但它也在提醒你：盒子的钱要另算。')

add('h3', '3.3 换算成人民币')
add('p', '成本只能在原口径里看，三组数据各管各的，别混算：')
add('bullets', [
    '**Terminal-Bench 2.1，全 Opus 跑一遍**：$98.06 / 267 次尝试，平均约 $0.37/次，==约人民币 2.6 元/次==。',
    '**LangChain 轻量任务**：每个完成任务约 $0.092，==约人民币 0.65 元/任务==。',
    '**另一个结构示例**：纯本地 $0、端云协同约 $0.42、纯云端约 $0.65——但材料没写清任务口径，不能和上面两组混算。',
])

add('h3', '3.4 落地三步走：先看清，再设门，最后调优')
add('img', 'meme_great.png', 2.2)
add('p', '**第一步：先把账本和路由看清。** 记录总推理 Token、云端出站 Token、云端计费 Token、云端剩余比例和任务质量，不急着冲最高本地化率。按场景建立「每次都上云」的基线，再引入本地分类与升级策略。这一步不是证明本地万能，而是看清哪些轮次真的需要云端。')
add('p', '**第二步：加 PII 网关和用户批准。** 所有出站流量收口到强制网关，接入实体级 PII 审查，展示确切出站内容，逐次批准。同时盯隐私检测效果、用户驳回率、误拦截率和额外交互延迟——安全机制不能只活在架构图里。')
add('p', '**第三步：上隔离子代理，按场景调阈值。** 给涉密计算加本地隔离子代理和 fail-closed，再分别调编码、客服、办公助理、工作流自动化的路由阈值。模型按设备和任务换成更大的单模型或多个小模型都行；选择标准回到任务成功率、云端占比、隐私指标和全链路延迟，而不是参数量标签。')

add('h3', '3.5 验收看板：别只盯「省了百分之几」')
add('p', '建议把评测分成四组，少一组都可能得到偏科答案：')
add('table', dict(cols=[1900, 4300, 5020], header=['维度', '核心指标', '要回答的问题'], rows=[
    ['**成本与流量**', '总推理 Token、云端出站 Token、云端计费 Token、云端账单节省', '到底是计算迁移了，还是云端账单真的降了？'],
    ['**任务质量**', '任务成功率、准确率差值', '省钱是不是拿不可接受的质量下降换的？'],
    ['**隐私安全**', 'PII recall、字符级 F1、用户驳回率、误拦截率', '该拦的拦住没有，不该拦的有没有过度阻断？'],
    ['**使用体验**', '全链路延迟', '本地路由、审查、批准和升级叠在一起，还能不能用？'],
]))
add('p', '评测尽量放在同一 harness、同一任务口径下比。尤其「全本地云账单节省 100%（推算）」只说明不再产生云端账单，推不出质量和云端方案一样。没在同一 harness 验证过，就老老实实写「待验证」，别把路线图写成战报。')

add('h3', '3.6 边界与风险：不是免费午餐，也不是万能防护罩')
add('bullets', [
    '**总成本边界**：本文的节省对象是云端账单，本地硬件与电费另计。云账单降了，不等于系统总拥有成本已被证明同比降了。',
    '**质量边界**：不同场景有 -0.3、-2.8 分、-6 分等质量变化，也有「持平」记录；它们来自不同数据来源和评测条件，不能横向拼成一句结论。',
    '**复现边界**：部分参数公开、部分任务集内部、部分配置未公开，企业域路由数据不可复现。证据等级不同，决策权重也该不同。',
    '**全本地边界**：云端剩余 0%、云端账单节省 100%，可以是推算或本方实测，但精度没在同一 harness 验证，就只能说明路线可能性。',
    '**隐私模型边界**：PII 审查有材料指标，但不是「零漏检」承诺；recall、F1、重复检出一致性和用户驳回要一起盯。',
    '**可用性边界**：fail-closed 会在沙箱不可用时禁用涉密能力，这是有意的安全取舍。组织得接受「宁可不执行，也不无保护执行」的产品行为。',
    '**模型绑定风险**：27B 和几个候选模型只是示意。把架构绑到某个名字上，会把「能力可替换」误做成「型号依赖」。',
])
add('hr')

# ---------------- 结论 ----------------
add('h2', '结论：让云端更贵重，而不是更常驻')
add('p', '端云协同最值钱的变化，不是把一朵云塞进一台设备，而是重新划分默认权力：本地先处理、本地先判断、本地先保护；云端只有在必要且获批时，才看到最小必要信息，给一句专家建议。')
add('p', '从成本看，减的是云端出站 Token 与云端计费 Token，不是宣称总 Token 消失；从架构看，云端从默认执行者变成按需升级的专家；从隐私看，0.6B PII 审查 + 逐次批准是第一道门，本地隔离子代理 + fail-closed 是第二道门，再用云端无工具权、无文件权、无完整会话权把最小暴露钉死。')
add('p', '对技术决策者来说，稳妥的起点不是先喊「省 80%」，而是先立统一 harness 和三本 Token 账，再按阶段上路由、隐私网关和隔离子代理。最终要证明的不是「本地模型很强」，而是：**在质量、延迟和隐私边界可接受的前提下，到底有多少活儿不必再让云端专家全程坐班。**')
add('hr')

# ---------------- 附 ----------------
add('h2', '附：数据来源')
add('p', '本文全部事实与数字，均整理自《端云协同省 Token 与隐私报告：写作事实本》所列内容：')
add('bullets', [
    '[《边缘AI·惊喜开箱3：视频生成省钱计划》](https://alidocs.dingtalk.com/i/nodes/yQod3RxJKGdGo75bIlMyPz7aJkb4Mw9r)',
    '[《端云协同省token》](https://alidocs.dingtalk.com/i/nodes/R4GpnMqJzG3G1EN5ILZd5nzz8Ke0xjE3)',
    '[《汇报材料》](https://alidocs.dingtalk.com/i/nodes/dxXB52LJqnLn1AEpsZqxwP7r8qjMp697)',
    '[《NCP-AgentApp-端云协同》HTML](https://alidocs.dingtalk.com/i/nodes/np9zOoBVBYwYok09SeyOOj59W1DK0g6l)',
    '[《portable_computer》HTML](https://alidocs.dingtalk.com/i/nodes/DnRL6jAJMGdGo0lxI9R7e2ABWyMoPYe1)',
], size=20)
add('p', '第二章的参考架构与第三章的官方口径对照：')
add('bullets', [
    'Perplexity Portable Computer 产品页：[perplexity.ai/hub/products/portable-computer](https://www.perplexity.ai/hub/products/portable-computer)；Perplexity Computer 发布博客：[perplexity.ai/hub/blog/introducing-perplexity-computer](https://www.perplexity.ai/hub/blog/introducing-perplexity-computer)',
    'NVIDIA NeMo Switchyard 仓库（路由器说明、Terminal-Bench 2.1 结果表）：[github.com/NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)；NVIDIA 技术博客：[Route AI Agents Across Models with NVIDIA NeMo Switchyard](https://developer.nvidia.com/blog/route-ai-agent-workloads-across-models-with-nvidia-nemo-switchyard/)',
    'LangChain 实测（145 个多轮任务、升级路由）：[How many of your agent\'s calls actually need a frontier model?](https://www.langchain.com/blog/switchyard-agent-routing-benchmark)',
], size=20)
add('p', '**测算口径声明**：本文所有节省比例均按表格标注为官方、第三方、推算或本方实测；云端账单节省不等于总 Token 减少；跨来源、跨任务、未在同一 harness 验证的数据不作为确定性横向结论。', size=20)

# ----------------------------------------------------------------------------
# XML helpers
# ----------------------------------------------------------------------------
def esc(s): return _esc(s, {'"': '&quot;'})

rels = []          # (rId, type, target, external)
_rid = [10]
def new_rid():
    _rid[0] += 1
    return f'rId{_rid[0]}'

def rpr(b=False, color=None, hl=None, sz=None, i=False, u=False):
    o = []
    if b: o.append('<w:b w:val="1"/>')
    if i: o.append('<w:i w:val="1"/>')
    if u: o.append('<w:u w:val="single"/>')
    if color: o.append(f'<w:color w:val="{color}"/>')
    if sz: o.append(f'<w:sz w:val="{sz}"/><w:szCs w:val="{sz}"/>')
    if hl: o.append(f'<w:highlight w:val="{hl}"/>')
    return f'<w:rPr>{"".join(o)}</w:rPr>' if o else ''

def run(text, **kw):
    return f'<w:r>{rpr(**kw)}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'

def hyperlink(text, url, sz=None):
    rid = new_rid()
    rels.append((rid, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink', url, True))
    return f'<w:hyperlink r:id="{rid}">{run(text, color=BLUE, u=True, sz=sz)}</w:hyperlink>'

TOKEN = re.compile(r'(\*\*.+?\*\*|==.+?==|\[[^\]]+?\]\([^)]+?\))')
def runs(markup, base=None):
    base = dict(base or {})
    out = []
    for part in TOKEN.split(markup):
        if not part: continue
        if part.startswith('**'):
            out.append(run(part[2:-2], **{**base, 'b': True}))
        elif part.startswith('=='):
            out.append(run(part[2:-2], **{**base, 'b': True, 'hl': 'yellow'}))
        elif part.startswith('['):
            m = re.match(r'\[(.+?)\]\((.+?)\)', part)
            out.append(hyperlink(m.group(1), m.group(2), sz=base.get('sz')))
        else:
            out.append(run(part, **base))
    return ''.join(out)

_bm = [100]
def para(inner, style=None, ppr='', bookmark=False):
    ps = f'<w:pStyle w:val="{style}"/>' if style else ''
    body = f'<w:pPr>{ps}{ppr}</w:pPr>{inner}'
    if bookmark:
        _bm[0] += 1
        body = f'<w:bookmarkStart w:id="{_bm[0]}" w:name="_h{_bm[0]}"/>{body}<w:bookmarkEnd w:id="{_bm[0]}"/>'
    return f'<w:p>{body}</w:p>'

_img_id = [0]
def image_run(path, width_twips):
    with open(path, 'rb') as f:
        head = f.read(32)
    w, h = struct.unpack('>II', head[16:24])   # PNG IHDR
    cx = width_twips * EMU_PER_TWIP
    cy = int(cx * h / w)
    rid = new_rid()
    name = os.path.basename(path)
    rels.append((rid, 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image', f'media/{name}', False))
    _img_id[0] += 1
    i = _img_id[0]
    return (f'<w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
            f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            f'<wp:docPr id="{i}" name="{name}"/>'
            f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
            f'<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            f'<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            f'<pic:nvPicPr><pic:cNvPr id="{i}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
            f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
            f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic>'
            f'</a:graphicData></a:graphic></wp:inline></w:drawing></w:r>'), name

media = []   # (name, path)

# numbering: abstract 0 = decimal (from B), abstract 1 = bullet; one num per list
nums = []    # abstractNumId per numId
def new_num(kind):
    nums.append(1 if kind == 'bullet' else 0)
    return len(nums)

BORDER = '<w:{side} w:val="single" w:sz="4" w:space="0" w:color="DADDE3"/>'
def table_xml(cols, header, rows, size=20, hl_rows=()):
    total = sum(cols)
    borders = ''.join(BORDER.format(side=s) for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
    tblpr = (f'<w:tblPr><w:tblStyle w:val="TableGrid"/><w:tblW w:w="{total}" w:type="dxa"/>'
             f'<w:tblBorders>{borders}</w:tblBorders><w:tblLayout w:type="fixed"/>'
             f'<w:tblCellMar><w:top w:w="70" w:type="dxa"/><w:left w:w="110" w:type="dxa"/><w:bottom w:w="70" w:type="dxa"/><w:right w:w="110" w:type="dxa"/></w:tblCellMar>'
             f'<w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/></w:tblPr>')
    grid = '<w:tblGrid>' + ''.join(f'<w:gridCol w:w="{w}"/>' for w in cols) + '</w:tblGrid>'
    def cell(text, w, shade=None, bold=False):
        shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{shade}"/>' if shade else ''
        base = {'sz': size}
        if bold: base['b'] = True
        p = para(runs(text, base), ppr='<w:spacing w:before="0" w:after="0"/>')
        return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/>{shd}</w:tcPr>{p}</w:tc>'
    trs = ['<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>' + ''.join(cell(t, w, 'F2F3F5', True) for t, w in zip(header, cols)) + '</w:tr>']
    for ri, r in enumerate(rows):
        sh = 'FDF2E9' if ri in hl_rows else None
        trs.append('<w:tr><w:trPr><w:cantSplit/></w:trPr>' + ''.join(cell(t, w, sh, ri in hl_rows) for t, w in zip(r, cols)) + '</w:tr>')
    return f'<w:tbl>{tblpr}{grid}{"".join(trs)}</w:tbl>' + para('', ppr='<w:spacing w:before="0" w:after="0"/><w:rPr><w:sz w:val="8"/></w:rPr>')

def toc_xml(entries):
    inner_w = TEXT_W - 2 * 360
    tab = f'<w:tabs><w:tab w:val="right" w:leader="dot" w:pos="{inner_w - 60}"/></w:tabs>'
    ps = [para(run('目录', b=True, sz=30), ppr='<w:spacing w:before="120" w:after="240"/><w:jc w:val="center"/>')]
    for h1, subs in entries:
        ps.append(para(run(h1, b=True, i=True, sz=22) + '<w:r><w:rPr><w:color w:val="C0C4CC"/></w:rPr><w:tab/></w:r>',
                       ppr=f'{tab}<w:spacing w:before="140" w:after="60"/>'))
        for s in subs:
            ps.append(para(run(s, color='5A6070', sz=20) + '<w:r><w:rPr><w:color w:val="C0C4CC"/></w:rPr><w:tab/></w:r>',
                           ppr=f'{tab}<w:spacing w:before="40" w:after="40"/><w:ind w:left="480"/>'))
    none = ''.join(f'<w:{s} w:val="nil"/>' for s in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'))
    tblpr = (f'<w:tblPr><w:tblW w:w="{TEXT_W}" w:type="dxa"/><w:tblBorders>{none}</w:tblBorders><w:tblLayout w:type="fixed"/>'
             f'<w:tblCellMar><w:top w:w="240" w:type="dxa"/><w:left w:w="360" w:type="dxa"/><w:bottom w:w="280" w:type="dxa"/><w:right w:w="360" w:type="dxa"/></w:tblCellMar></w:tblPr>')
    tc = f'<w:tc><w:tcPr><w:tcW w:w="{TEXT_W}" w:type="dxa"/><w:shd w:val="clear" w:color="auto" w:fill="F5F6F8"/></w:tcPr>{"".join(ps)}</w:tc>'
    return (f'<w:tbl>{tblpr}<w:tblGrid><w:gridCol w:w="{TEXT_W}"/></w:tblGrid><w:tr>{tc}</w:tr></w:tbl>'
            + para('', ppr='<w:spacing w:before="0" w:after="120"/>'))

HR = '<w:p><w:pPr><w:pBdr><w:top w:val="single" w:sz="6" w:space="1" w:color="E3E5E8"/></w:pBdr><w:spacing w:before="200" w:after="200"/></w:pPr></w:p>'

# ----------------------------------------------------------------------------
# build body
# ----------------------------------------------------------------------------
body = []
for kind, args, opts in DOC:
    base = {'sz': opts['size']} if 'size' in opts else {}
    if kind == 'title':
        body.append(para(run(args[0], b=True, sz=48), style='dingding-heading1'))
    elif kind == 'tag':
        body.append(para(runs(args[0], {'b': True, 'color': BLUE}), ppr='<w:spacing w:before="60" w:after="160"/>'))
    elif kind == 'toc':
        body.append(toc_xml(args[0]))
    elif kind == 'h2':
        body.append(para(run(args[0]), style='dingding-heading2', bookmark=True))
    elif kind == 'h3':
        body.append(para(run(args[0]), style='dingding-heading3', bookmark=True))
    elif kind == 'p':
        body.append(para(runs(args[0], base), ppr='<w:spacing w:before="0" w:after="160"/>'))
    elif kind == 'quote':
        body.append(para(runs(args[0]), style='dingding_quote', ppr='<w:spacing w:before="0" w:after="160"/>'))
    elif kind == 'bullets' or kind == 'numbered':
        nid = new_num('bullet' if kind == 'bullets' else 'decimal')
        for item in args[0]:
            body.append(para(runs(item, base), ppr=f'<w:numPr><w:ilvl w:val="0"/><w:numId w:val="{nid}"/></w:numPr><w:spacing w:before="0" w:after="80"/>'))
        body.append(para('', ppr='<w:spacing w:before="0" w:after="60"/>'))
    elif kind == 'table':
        t = args[0]
        body.append(table_xml(t['cols'], t['header'], t['rows'], size=t.get('size', 20), hl_rows=t.get('hl_rows', ())))
    elif kind == 'img':
        path = os.path.join(HERE, args[0])
        width = TEXT_W if args[1] == 'full' else int(args[1] * 1440)
        r, name = image_run(path, width)
        media.append((name, path))
        body.append(para(r, ppr='<w:spacing w:before="120" w:after="200"/><w:jc w:val="center"/>'))
    elif kind == 'hr':
        body.append(HR)
    else:
        raise ValueError(kind)


SECT = '<w:sectPr><w:pgSz w:w="13380" w:h="16905"/><w:pgMar w:top="720" w:right="1080" w:bottom="720" w:left="1080" w:header="851" w:footer="992" w:gutter="0"/><w:type w:val="nextPage"/></w:sectPr>'

# ----------------------------------------------------------------------------
# assemble package
# ----------------------------------------------------------------------------
work = os.path.join(HERE, 'pkg')
if os.path.exists(work): shutil.rmtree(work)
with zipfile.ZipFile(SRC_DOCX) as z: z.extractall(work)

orig = open(os.path.join(work, 'word/document.xml'), encoding='utf8').read()
root_open = orig[:orig.index('<w:body>')]
doc_xml = root_open + '<w:body>' + ''.join(body) + SECT + '</w:body></w:document>'
open(os.path.join(work, 'word/document.xml'), 'w', encoding='utf8').write(doc_xml)

# numbering.xml
numb = open(os.path.join(work, 'word/numbering.xml'), encoding='utf8').read()
head = numb[:numb.index('<w:abstractNum ')]
abs0 = ('<w:abstractNum w:abstractNumId="0" w15:restartNumberingAfterBreak="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="420" w:hanging="420"/></w:pPr><w:rPr/></w:lvl></w:abstractNum>')
abs1 = ('<w:abstractNum w:abstractNumId="1" w15:restartNumberingAfterBreak="0"><w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="bullet"/><w:lvlText w:val="•"/><w:lvlJc w:val="left"/><w:pPr><w:ind w:left="420" w:hanging="420"/></w:pPr><w:rPr/></w:lvl></w:abstractNum>')
numxml = ''.join(f'<w:num w:numId="{i + 1}"><w:abstractNumId w:val="{a}"/></w:num>' for i, a in enumerate(nums))
open(os.path.join(work, 'word/numbering.xml'), 'w', encoding='utf8').write(head + abs0 + abs1 + numxml + '</w:numbering>')

# rels
os.makedirs(os.path.join(work, 'word/media'), exist_ok=True)
for name, path in media:
    shutil.copy(path, os.path.join(work, 'word/media', name))
relx = open(os.path.join(work, 'word/_rels/document.xml.rels'), encoding='utf8').read()
extra = ''.join(f'<Relationship Id="{rid}" Type="{t}" Target="{esc(tg)}"' + (' TargetMode="External"' if ext else '') + '/>'
                for rid, t, tg, ext in rels)
relx = relx.replace('</Relationships>', extra + '</Relationships>')
open(os.path.join(work, 'word/_rels/document.xml.rels'), 'w', encoding='utf8').write(relx)

# docProps title
core = os.path.join(work, 'docProps/core.xml')
if os.path.exists(core):
    c = open(core, encoding='utf8').read()
    c = re.sub(r'<dc:title>.*?</dc:title>', '<dc:title>边缘AI·惊喜开箱4：端云协同省钱计划</dc:title>', c, flags=re.S)
    open(core, 'w', encoding='utf8').write(c)

if os.path.exists(OUT_DOCX): os.remove(OUT_DOCX)
with zipfile.ZipFile(OUT_DOCX, 'w', zipfile.ZIP_DEFLATED) as z:
    # [Content_Types].xml first
    z.write(os.path.join(work, '[Content_Types].xml'), '[Content_Types].xml')
    for dp, dn, fn in os.walk(work):
        for f in fn:
            full = os.path.join(dp, f)
            arc = os.path.relpath(full, work)
            if arc == '[Content_Types].xml': continue
            z.write(full, arc)
print('wrote', OUT_DOCX, 'blocks', len(DOC), 'images', len(media), 'lists', len(nums), 'rels', len(rels))
