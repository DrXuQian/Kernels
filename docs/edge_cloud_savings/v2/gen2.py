# -*- coding: utf-8 -*-
"""Generate the rewritten DingTalk-style docx from B's package + new content."""
import os, re, shutil, zipfile, struct, sys
from xml.sax.saxutils import escape as _esc

SRC_DOCX = sys.argv[1]          # original B.docx
OUT_DOCX = sys.argv[2]          # output path
HERE = os.path.dirname(os.path.abspath(__file__))

BLUE = "1E6FD9"
GREY = "8A8F99"
TEXT_W = 14745                  # twips: 16905 - 2*1080
EMU_PER_TWIP = 635

# ----------------------------------------------------------------------------
# content
# ----------------------------------------------------------------------------
DOC = []
def add(kind, *args, **kw): DOC.append((kind, args, kw))

add('title', '别让每个 Token 都上云：端云协同的省钱与隐私计划')
add('tag', '别让每个 Token 都上云。端侧 AI Box 不只是一台跑模型的盒子，更是给你的 Agent 配的一位【本地店长】：日常的活儿自己干，真遇到疑难杂症，再按铃叫云端专家。数据不出门，账单变薄。')
add('tag', '花十分钟读完，你会🉐到三件事：第一章，数据不出门是怎么做到的，你在哪里能看见它，以及为什么这事比省钱还靠前；第二章，云端账单到底能省多少；第三章，这套东西长什么样，照着谁搭。')

TOC = [
    ('一、隐私：数据不出门，才敢把活儿交给 AI', ['1.1 为什么隐私排第一', '1.2 隐私是怎么保住的：一道门、一个门卫、一间小黑屋', '1.3 隐私在哪里看得见']),
    ('二、省钱：云端账单到底能省多少', ['2.1 先说清省的是哪本账', '2.2 钱是怎么省下来的：三次截流', '2.3 各场景能省多少', '2.4 换算成人民币', '2.5 别被数字忽悠：四条提醒']),
    ('三、架构：本地当店长，云端做会诊', ['3.1 抄谁的作业：Perplexity 管「家」，Switchyard 管「路」', '3.2 一图看完：五步闭环', '3.3 谁干什么、不许干什么', '3.4 怎么落地、怎么验收']),
    ('结论：让云端更贵重，而不是更常驻', []),
    ('附一：分场景完整数据', []),
    ('附二：数据来源', []),
]
add('toc', TOC)

# ================= 一、隐私 =================
add('h2', '一、隐私：数据不出门，才敢把活儿交给 AI')
add('img', 'meme_all.png', 2.6)
add('p', '很多人一听「端云协同」，第一反应是省钱。其实顺序反了：**先是数据不出门，然后才谈账单。** 省钱是加分项，隐私是准入门槛。省钱和隐私也不用二选一，这一章先讲怎么把门守住。')

add('h3', '1.1 为什么隐私排第一')
add('p', 'AI Agent 和聊天机器人不是一回事。聊天机器人只看你打的那几句话；Agent 要替你干活，就得看你的文件、邮件、代码、客户资料、聊天记录。全部上云的方案，等于每接一个任务，就把半块硬盘和公司通讯录打包寄出去一次。')
add('p', '寄出去之后会发生什么，你其实不知道。这就是纯云方案最让人不踏实的地方：')
add('bullets', [
    '**你不知道送了什么。** 模型「觉得有用」的上下文都会被打包，里面有没有客户电话、有没有合同金额，没人替你检查。',
    '**你不知道谁在看。** 服务条款说不训练、不留存，你只能选择相信。',
    '**出事没法收回。** 一份客户名单泄露出去，不会因为你后来关掉了功能而回来。',
])
add('p', '所以对企业来说，「能不能把 Agent 用起来」这个问题，第一道关卡不是效果，而是**敢不敢让它看东西。** 本地优先的意义就在这：让它看，但不让它带出门。')

add('h3', '1.2 隐私是怎么保住的：一道门、一个门卫、一间小黑屋')
add('p', '只在系统提示里写一句「请注意隐私」，就像在门口贴张纸条「贵重物品请勿带出」。有礼貌，没用。我们要的是真的会关上的门。整套做法用三个词就能记住：')
add('p', '**一道门。** 整台盒子默认零出网，只留一个网络出口，而且只有 Agent 内置的「出网门控」能走这个口。任何工具、任何沙箱都不能自己联网。想出门，只能过门控。')
add('p', '**一个门卫。** 门控里常驻一个 0.6B 的小模型，专门干一件事：把要出门的内容逐字扫一遍，看有没有隐私。它认 9 类东西：人名、邮箱、电话、地址、网址、日期、账号、机密事项和其他。抓到了，要么打码，要么拦下。扫完之后，系统把「即将离开设备的确切内容」弹给你看，你点头，这一小段才发。批准的是这一次、这一段，不是给整台设备发通行证。发出去的目标也只能是白名单里的地址，每一次都留账。')
add('p', '**一间小黑屋。** 真正涉密的计算，比如算工资、看合同、翻客户数据，连门都不让它靠近。这类活儿派给一个隔离的本地子代理，在自己的沙箱里算完，只回一句结论，不回原始数据。云端要是需要参考，拿到的也只是「非身份汇总」。')
add('p', '还有一条最关键的规矩：**门锁坏了，门就一直关着。** 沙箱不可用，涉密功能直接禁用，不做「先跑完再说」的降级。很多系统的坏习惯是安全组件挂了照样干活，这里反过来。')
add('p', '云端那边呢？它碰不到本地文件，调不了本地工具，看不到完整对话，也不能主动往回推东西。它只收到你批准过的那一小段，只能回一段文字建议。就算叫了云端，暴露面也只有这一次任务的最小集合。')

add('h3', '1.3 隐私在哪里看得见')
add('p', '安全机制最怕只活在架构图里。这套方案里，隐私是用户每天都摸得到的东西：')
add('bullets', [
    '**出门前的弹窗。** 每次要上云，你先看到要发的原文，敏感字段已经高亮或打码。不点，就不发。',
    '**一本流水账。** 什么时候、发了什么、发去哪、谁批的，全有记录。审计不用靠回忆。',
    '**门卫的成绩单。** 材料里的实测：长文档里 100 处隐私能抓到 97 处（原来是 83 处）；同一个人名反复出现时，能一直认出来的比例是 79.4%，对比前沿云端大模型只有 57.0%；而且它和主模型共用一张卡，显存只占 5%。',
    '**拒绝也算数据。** 用户驳回率、误拦截率会一直被盯着。拦得太松是漏，拦得太紧是烦，两头都要调。',
])
add('p', '对比一下纯云方案：以上四样，一样都没有。你只能相信服务条款。')
add('quote', '这些指标来自材料中的设计与评测记录，落地时仍要在自己的数据和链路上复验。PII 审查不是「零漏检」承诺。')
add('hr')

# ================= 二、省钱 =================
add('h2', '二、省钱：云端账单到底能省多少')
add('img', 'meme_money.png', 2.4)

add('h3', '2.1 先说清省的是哪本账')
add('p', '一聊「省 Token」，很多人脑子里只有一本账。其实至少有三本：')
add('table', dict(cols=[3300, 5000, 6445], header=['哪本账', '它在问什么', '端云协同之后'], rows=[
    ['**总推理 Token**', '本地 + 云端一共嚼了多少内容', '不一定变少。本地做规划、分类、汇总，同样在吐 Token'],
    ['**云端出站 Token**', '有多少内容真的离开了设备', '变少。本地先干、上下文裁剪、最小暴露，三招都在压它'],
    ['**云端计费 Token**', '有多少 Token 进了云端的收银台', '变少。上云次数少了，每次送的也少了，账单直接瘦身'],
]))
add('p', '所以准确的说法不是「Token 凭空消失」，而是：**把原本默认送云的活儿留在本地，把必须上云的那部分精简后再送。** Token 没少，只是换了一家收银台。本地这家不按 Token 收钱，只收电费。')
add('p', '本文所有「省 X%」都拿同一把尺子量：**每一次调用都走 Opus**。本地模型只算电费和硬件，不进云端账单。这把尺子擅长回答「云账单能少多少」，回答不了「系统总成本一定少多少」。盒子的钱在另一张账上，先说清楚，后面的数字才不会被读歪。')

add('h3', '2.2 钱是怎么省下来的：三次截流')
add('p', '**第一次：本地直接干完。** 一轮任务不需要上云，云端就少收一轮的钱。')
add('p', '**第二次：只把难点送上去。** 一个多轮任务不必从头到尾绑在云端，可以按轮、按阶段路由。云端从「全程包办」变成「疑难会诊」。')
add('p', '**第三次：只送最小必要的那一段。** 即便上云，也不把本地文件、完整对话、全部工具状态一股脑发出去，而是本地裁剪、汇总后再出站。这一刀和第一章的隐私门卫是同一刀：少送就是少花，也是少泄露。')
add('p', '真正决定能省多少的，不是「有没有本地模型」，而是本地模型能独立扛住多少轮。这也是下面各场景差得这么多的原因：编码最难扛，客服最好扛。')

add('h3', '2.3 各场景能省多少')
add('img', 'fig_savings.png', 'full')
add('p', '把上图压成一句话：**编码省四到五成，办公助理省七到八成，客服多轮省七五到八五。** 完整表格在附一，这里只列最该记住的几行：')
add('table', dict(cols=[3300, 2600, 2900, 5945], size=20, header=['场景', '还要上云的比例', '云端账单能省', '效果怎么样'], rows=[
    ['编码 / 运维（Terminal-Bench 2.1）', '51–61%', '**40–50%**', '准确率 75.7% 对 76.0%，只差 0.3。NVIDIA 官方数据，可复现'],
    ['办公助理 / 工作流（LangChain 145 个任务）', '约 18–27%', '**70–82%**', '持平到 -6 分。第三方实测 + NVIDIA 内部基准'],
    ['客服多轮（τ²-bench，激进档）', '15–25%', '**75–85%**', '解题率 0.891 对 0.903。NVIDIA 标定，可复现'],
    ['全本地（强模型也放盒子里）', '0%', '100%', '效果待同一评测框架验证，先当路线图，不当数字'],
]))
add('p', '下面这张是 NVIDIA 自己的账本，看趋势就够了：横轴是完成任务的花费，纵轴是完成率。全走 Opus 4.8 在最右边，用 Switchyard 把大部分活儿路由给小模型之后，完成率几乎不动，花费只剩不到三分之一。')
add('img', 'nv_switchyard.png', 'full')
add('quote', '图片来源：NVIDIA 博客《NVIDIA Nemotron 3.5 Lightning and NeMo Switchyard Deliver Faster, Smarter, More Efficient Agentic AI》。注意它把本地那个小模型也按 API 单价计了钱，所以省得比本文口径少；本文假设小模型跑在自己的盒子上，只数云端那部分。')

add('h3', '2.4 换算成人民币')
add('p', '三组数据各管各的，别混算：')
add('bullets', [
    '**编码任务，全走 Opus：** Terminal-Bench 2.1 跑一遍 $98.06，267 次尝试，平均 ==约 2.6 元一次==。省一半就是每次省一块三。',
    '**办公助理类轻量任务：** 每完成一个任务约 $0.092，==约 0.65 元==。省七八成之后，一个任务一毛多。',
    '**另一个结构示例：** 纯本地 $0、端云协同约 $0.42、纯云端约 $0.65。材料没写清任务口径，只能看比例，不能和上面两组混算。',
])

add('h3', '2.5 别被数字忽悠：四条提醒')
add('bullets', [
    '**省的是云账单，不是总成本。** 盒子的硬件和电费另算。云账单降了，不等于总拥有成本已经被证明降了。',
    '**效果有代价，代价不一样。** 有的场景只差 0.3 分，有的差 6 分。它们来自不同的评测，不能拼成一句「效果不变」。',
    '**证据有等级。** 有的可复现，有的评测集在别人手里，有的配置没公开。附一每一行都标了来源，决策时按证据等级给权重。',
    '**全本地 100% 只说明不花云端的钱。** 效果好不好，得在同一个评测框架里跑过才算数。没跑过，就老实写「待验证」。',
])
add('hr')

# ================= 三、架构 =================
add('h2', '三、架构：本地当店长，云端做会诊')
add('p', '前两章说的「门」和「截流」，落到一台设备上是什么样？这一章讲清楚。好消息是不用从零画图，业界已经有两份现成的作业。')

add('h3', '3.1 抄谁的作业：Perplexity 管「家」，Switchyard 管「路」')
add('p', '**Perplexity Portable Computer，负责「家里怎么布置」。** 2026 年 2 月 Perplexity 发布了 Computer：一个会拆任务、派子代理、在沙箱里跑工具、还接 Slack / GitHub / 邮件的多模型 Agent。8 月它又和 NVIDIA 一起推出了完全本地版 Portable Computer：编排模型、子代理模型、Agent 运行时、工具沙箱、连接器，全部跑在你自己的硬件上（DGX Spark，或任何 24GB 显存以上的 RTX 卡），本地模型是一个专门为 Agent 后训练过的 27B。本地完成的步骤**不消耗一分钱云端额度**；真需要前沿推理时，先弹窗要你批准，出站内容先过隐私标记，云端只能回文字建议。沙箱是操作系统级强制隔离，沙箱不可用就直接禁用工具执行。第一章那三个词，基本就是从这来的。')
add('p', '**NVIDIA NeMo Switchyard，负责「什么时候出门」。** 它是 NVIDIA 开源的模型路由层，专门回答一个问题：Agent 的这一次调用，到底要不要花前沿模型的钱？它自带几种现成的路由器，大白话版：')
add('bullets', [
    '**升级路由：** 先让便宜模型干，旁边有个裁判盯着。连续两次判定「卡住了」，这个任务之后全交给贵模型。只升不降。',
    '**阶段路由：** 看最近的进度。稳稳当当改代码、测试都过了，用便宜的；报错连连、原地打转，换贵的。',
    '**分类路由：** 一开始就判断这活儿难不难，定了模型就不再反复换。',
    '**可训练路由：** 学着预测哪个模型更可能做对，再按成本和延迟做取舍。',
])
add('p', '第二章那些「还要上云 X%」，大多就是这些路由器在 Terminal-Bench、τ²-bench 和 LangChain 任务集上跑出来的。一句话：**Perplexity 告诉你本地这个家怎么布置，Switchyard 告诉你什么时候该出门、出门带多少东西。**')

add('h3', '3.2 一图看完：五步闭环')
add('img', 'meme_tutu.png', 2.6)
add('img', 'fig_arch.png', 'full')
add('p', '图里三种颜色对应三块地盘：GPU 区常驻两个模型（27B 主推理 + 0.6B 隐私门卫），CPU 区跑 Agent 运行时、子代理和两套沙箱，右边橙色是受控访问的云端。整台设备只留一个网络出口。一个任务从左到右走五步：')
add('p', '**第一步：你下达任务。** 请求先进本地编排器。会话、本地文件、任务上下文默认留在设备上。')
add('p', '**第二步：本地模型先上。** 27B 主模型负责规划、分类和编排；能本地完成的直接完成，需要工具就交给本地沙箱，沙箱零出网。')
add('p', '**第三步：确需云端，先过门卫。** 先裁剪上下文，只留任务需要的最小片段；再过 0.6B 隐私审查；然后弹窗给你看，等你点头。')
add('p', '**第四步：云端专家按需处理。** 你批准了，云端模型群才上场。它只收到批准过的片段，只能回一段文字建议。')
add('p', '**第五步：结果回本地继续跑。** 云端输出不接管系统，而是回到本地编排器，由本地模型结合本地状态和工具继续执行。')
add('p', '所以云端是「按铃才来的会诊专家」，不是一直坐在驾驶座上的默认司机。')

add('h3', '3.3 谁干什么、不许干什么')
add('table', dict(cols=[3400, 5800, 5545], header=['组件', '主要职责', '明确边界'], rows=[
    ['本地编排器 / 主模型', '规划、分类、路由、常规推理、上下文整理', '规格可替换，不绑定 27B'],
    ['本地工具沙箱', '执行工具与任务动作', '隔离环境、零出网'],
    ['0.6B 隐私审查模型', '对出站内容做逐字检测', '常驻本地，规格为材料示意'],
    ['本地隔离子代理', '处理涉密计算，只回传非身份汇总', '沙箱不可用则禁用，不降级裸奔'],
    ['云端模型群', '本地啃不动的前沿推理、实时数据类任务', '无本地工具权、文件权和完整会话权'],
]))
add('p', '顺便说一句型号：文中的 27B 主模型、0.6B 隐私模型、256K 上下文，以及 Qwen3.8-Flash、DeepSeek V4 Flash 这些名字，都是**架构示意或候选替换**，不是采购清单。27B 可以换成更大的单模型，也可以换成几个小模型搭班子。你要决策的是「能力怎么分工」，不是「买哪个型号」。')

add('h3', '3.4 怎么落地、怎么验收')
add('img', 'meme_great.png', 2.2)
add('p', '**第一步：先看清账本。** 记录三本 Token 账、还要上云的比例和任务质量，不急着冲最高本地化率。按场景先跑一遍「每次都上云」的基线，再引入本地路由。这一步不是证明本地万能，而是看清哪些轮次真的需要云端。')
add('p', '**第二步：装门卫。** 所有出站流量收口到一个强制网关，接入隐私审查，展示确切出站内容，逐次批准。同时盯驳回率、误拦截率和多出来的等待时间。')
add('p', '**第三步：盖小黑屋，按场景调阈值。** 给涉密计算加隔离子代理和「门锁坏了就关门」的策略，再分别调编码、客服、办公助理的路由阈值。')
add('p', '验收看四组指标，少一组都会得到偏科答案：')
add('table', dict(cols=[2600, 5700, 6445], header=['看什么', '指标', '要回答的问题'], rows=[
    ['**成本与流量**', '三本 Token 账、云端账单节省', '到底是计算搬家了，还是云端账单真的降了？'],
    ['**任务质量**', '任务成功率、准确率差值', '省钱是不是拿不可接受的质量下降换的？'],
    ['**隐私安全**', '隐私召回率、用户驳回率、误拦截率', '该拦的拦住没有，不该拦的有没有过度阻断？'],
    ['**使用体验**', '全链路延迟', '路由、审查、批准、上云叠在一起，还能不能用？'],
]))
add('hr')

# ================= 结论 =================
add('h2', '结论：让云端更贵重，而不是更常驻')
add('p', '端云协同最值钱的变化，不是把一朵云塞进一台设备，而是重新划分默认权力：本地先处理、本地先判断、本地先保护；云端只有在必要且你点头时，才看到最小必要的那一段，给一句专家建议。')
add('p', '从隐私看，一道门、一个门卫、一间小黑屋，加上「门锁坏了就关门」，把数据钉在设备里；从成本看，减的是云端出站和云端计费，不是宣称 Token 消失；从架构看，云端从默认执行者变成按需升级的专家。')
add('p', '对技术决策者来说，稳妥的起点不是先喊「省 80%」，而是先立统一的评测框架和三本 Token 账，再按阶段上路由、隐私网关和隔离子代理。最终要证明的不是「本地模型很强」，而是：**在质量、延迟和隐私边界可接受的前提下，到底有多少活儿不必再让云端专家全程坐班。**')
add('hr')

# ================= 附一 =================
add('h2', '附一：分场景完整数据')
add('quote', '下表的「节省」都是相对「每次调用都走 Opus」的云端账单节省，不等于总推理 Token 减少，也没算本地硬件与电费。')
add('table', dict(cols=[2900, 2400, 1500, 1800, 2900, 3245], size=18,
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
add('p', '顺手解释一个常被问到的差异：Switchyard 官方账本里，本地那个 Nemotron 也是按 API 单价计费的，所以官方给的总成本降幅（比如 Terminal-Bench 2.1 上升级路由只便宜 13.3%）比本文的「云端账单节省」低。本文的口径是本地模型跑在自己的盒子上、只交电费，只数云端那部分。这是口径差，不是魔法。', size=20)
add('p', '**测算口径声明**：本文所有节省比例均按表格标注为官方、第三方、推算或本方实测；云端账单节省不等于总 Token 减少；跨来源、跨任务、未在同一 harness 验证的数据不作为确定性横向结论。', size=20)

# ================= 附二 =================
add('h2', '附二：数据来源')
add('p', '本文全部事实与数字，均整理自《端云协同省 Token 与隐私报告：写作事实本》所列内容：')
add('bullets', [
    '[《边缘AI·惊喜开箱3：视频生成省钱计划》](https://alidocs.dingtalk.com/i/nodes/yQod3RxJKGdGo75bIlMyPz7aJkb4Mw9r)',
    '[《端云协同省token》](https://alidocs.dingtalk.com/i/nodes/R4GpnMqJzG3G1EN5ILZd5nzz8Ke0xjE3)',
    '[《汇报材料》](https://alidocs.dingtalk.com/i/nodes/dxXB52LJqnLn1AEpsZqxwP7r8qjMp697)',
    '[《NCP-AgentApp-端云协同》HTML](https://alidocs.dingtalk.com/i/nodes/np9zOoBVBYwYok09SeyOOj59W1DK0g6l)',
    '[《portable_computer》HTML](https://alidocs.dingtalk.com/i/nodes/DnRL6jAJMGdGo0lxI9R7e2ABWyMoPYe1)',
], size=20)
add('p', '第三章的参考架构与第二章的官方口径对照：')
add('bullets', [
    'Perplexity Portable Computer 产品页：[perplexity.ai/hub/products/portable-computer](https://www.perplexity.ai/hub/products/portable-computer)；Perplexity Computer 发布博客：[perplexity.ai/hub/blog/introducing-perplexity-computer](https://www.perplexity.ai/hub/blog/introducing-perplexity-computer)',
    'NVIDIA 博客（第二章图片来源）：[NVIDIA Nemotron 3.5 Lightning and NeMo Switchyard Deliver Faster, Smarter, More Efficient Agentic AI](https://blogs.nvidia.com/blog/nemotron-lightning-switchyard-rtx-dgx/)',
    'NVIDIA NeMo Switchyard 仓库（路由器说明、Terminal-Bench 2.1 结果表）：[github.com/NVIDIA-NeMo/Switchyard](https://github.com/NVIDIA-NeMo/Switchyard)；NVIDIA 技术博客：[Route AI Agents Across Models with NVIDIA NeMo Switchyard](https://developer.nvidia.com/blog/route-ai-agent-workloads-across-models-with-nvidia-nemo-switchyard/)',
    'LangChain 实测（145 个多轮任务、升级路由）：[How many of your agent\'s calls actually need a frontier model?](https://www.langchain.com/blog/switchyard-agent-routing-benchmark)',
], size=20)

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


SECT = '<w:sectPr><w:pgSz w:w="16905" w:h="23820"/><w:pgMar w:top="720" w:right="1080" w:bottom="720" w:left="1080" w:header="851" w:footer="992" w:gutter="0"/><w:type w:val="nextPage"/></w:sectPr>'

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
import glob
for f in glob.glob(os.path.join(work, 'word/media/*')):
    if os.path.basename(f) not in [n for n, _ in media]: os.remove(f)
relx = open(os.path.join(work, 'word/_rels/document.xml.rels'), encoding='utf8').read()
relx = re.sub(r'<Relationship [^>]*Type="[^"]*/(?:image|hyperlink)"[^>]*/>\s*', '', relx)
extra = ''.join(f'<Relationship Id="{rid}" Type="{t}" Target="{esc(tg)}"' + (' TargetMode="External"' if ext else '') + '/>'
                for rid, t, tg, ext in rels)
relx = relx.replace('</Relationships>', extra + '</Relationships>')
open(os.path.join(work, 'word/_rels/document.xml.rels'), 'w', encoding='utf8').write(relx)

# docProps title
core = os.path.join(work, 'docProps/core.xml')
if os.path.exists(core):
    c = open(core, encoding='utf8').read()
    c = re.sub(r'<dc:title>.*?</dc:title>', '<dc:title>别让每个 Token 都上云：端云协同的省钱与隐私计划</dc:title>', c, flags=re.S)
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
