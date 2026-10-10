"""Streamlit demo: build a plan with the fixed pipeline, then keep it up to date by marking
questions and chatting with the controller agent.

Run locally:  streamlit run demo.py
If Qdrant Cloud is unreachable, use the local vector store:  QDRANT_URL= streamlit run demo.py
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from orchestration import DEFAULT_USER
from scripts.controller import graph
from scripts.controller.feedback import skill_factor
from scripts.controller.memory import get_preferences, recent_log
from scripts.controller.replan import today_index

st.set_page_config(page_title="DS Interview Copilot", layout="wide")

USER = DEFAULT_USER  # local demo: one user, no login
STEPS = {
    "agent1": "分析技能（Agent1）",
    "agent_scope": "规划范围（Scope Planner）",
    "agent2": "检索题目（Agent2）",
    "agent3": "排期（Agent3）",
}
TOOL_LABEL = {
    "update_plan": "调整计划",
    "retrieve_questions": "检索题目",
    "analyze_skills": "分析技能",
    "generate_plan": "重新生成计划",
}
DIFF_LABEL = {"easy": "简单", "medium": "中等", "hard": "困难"}
STATUS_ICON = {"done": ":material/check_circle:", "wrong": ":material/cancel:", "todo": ":material/radio_button_unchecked:"}
WELCOME = "计划已经生成。告诉我你的进展，比如哪些题做完了、哪道做错了、哪个方向不会，我来调整明天之后的安排。"

ss = st.session_state
ss.setdefault("thread_id", None)
ss.setdefault("new_ids", {})  # thread_id -> ids added to the plan by the last chat turn
ss.setdefault("flash", None)  # message to show as a toast after a rerun

if ss.flash:
    st.toast(ss.flash)
    ss.flash = None


def question_link(task: dict) -> str:
    # SQL questions link to the backup page; others prefer the original url.
    url = task.get("backup_url") if task.get("category") == "SQL" else (task.get("url") or task.get("backup_url"))
    return f"[{task['title']}]({url})" if url else task["title"]


# ---------- sidebar ----------
def render_memory(state: dict) -> None:
    user = state.get("user_id") or USER
    weights = state.get("weights") or {}
    weak = [s for s in weights if skill_factor(user, s) > 1]
    mastered = [s for s in weights if skill_factor(user, s) < 1]
    titles = {t["id"]: t["title"] for day in state.get("days") or [] for t in day}

    st.markdown("**长期记忆**")
    st.caption("薄弱技能")
    st.markdown(" ".join(f":orange-badge[{s}]" for s in weak) or "暂无")
    st.caption("已掌握")
    st.markdown(" ".join(f":blue-badge[{s}]" for s in mastered) or "暂无")
    st.caption("偏好")
    prefs = get_preferences(user) or {}
    st.markdown(f"每天最多 {prefs['daily_load']} 题" if prefs.get("daily_load") else "暂无")
    st.caption("最近记录")
    rows = recent_log(user, limit=5)
    if not rows:
        st.markdown("暂无")
    for qid, skill, result in rows:
        if result == "struggling":
            st.markdown(f":orange[自述薄弱] {skill}")
        else:
            label = ":blue[完成]" if result == "done" else ":orange[做错]"
            st.markdown(f"{label} {titles.get(qid, qid)}")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### DS Interview Copilot")
        st.caption("面试学习助手")
        if st.button("新计划", icon=":material/add:", type="primary", use_container_width=True):
            ss.thread_id = None
            st.rerun()

        st.markdown("**我的计划**")
        threads = graph.list_threads(USER)
        if not threads:
            st.caption("还没有计划")
        for th in threads:
            active = th["thread_id"] == ss.thread_id
            if st.button(f"{th['title']} · {th['days']} 天", key=f"thread-{th['thread_id']}",
                         type="secondary", icon=":material/arrow_right:" if active else None, use_container_width=True):
                ss.thread_id = th["thread_id"]
                st.rerun()

        if ss.thread_id:
            st.divider()
            render_memory(graph.get_state(ss.thread_id))
            st.divider()
            if st.button("模拟：进入下一天", icon=":material/skip_next:", use_container_width=True):
                index = graph.advance_day(ss.thread_id)
                ss.flash = f"已模拟进入第 {index + 1} 天"
                st.rerun()
            st.caption("仅用于演示：把计划的开始日期往前挪一天")


# ---------- new plan ----------
def render_new_plan() -> None:
    st.title("新建学习计划")
    st.caption("第一次生成计划走固定流水线；之后的调整交给对话中的 agent。")
    with st.form("new-plan"):
        title = st.text_input("计划名称（可选）", placeholder="比如：数据科学家 · 产品分析")
        jd = st.text_area("岗位描述（JD）", height=180, placeholder="粘贴岗位描述……")
        user_desc = st.text_area("你的背景", height=120, placeholder="你的背景、擅长和薄弱的方向、其他要求……")
        days_left = st.slider("距离面试的天数", min_value=1, max_value=15, value=7)
        submitted = st.form_submit_button("生成计划", type="primary")
    if not submitted:
        return
    if not jd.strip() or not user_desc.strip():
        st.warning("请填写岗位描述和你的背景。")
        return

    thread_id = str(uuid.uuid4())
    with st.status("正在生成计划……", expanded=True) as status:
        try:
            # The graph streams one update per finished pipeline node.
            for node in graph.stream_plan(jd, user_desc, days_left, thread_id, USER, title=title):
                st.write(f":material/check: {STEPS.get(node, node)}")
        except Exception as error:
            status.update(label="生成失败", state="error")
            st.error(f"{error}\n\n如果是 Qdrant Cloud 连接问题，可以用 `QDRANT_URL= streamlit run demo.py` 改用本地向量库。")
            return
        status.update(label="计划已生成", state="complete")
    ss.thread_id = thread_id
    st.rerun()


# ---------- plan ----------
def mark(thread_id: str, task: dict, result: str) -> None:
    graph.submit_feedback(thread_id, {task["id"]: result})
    ss.flash = (f"已记录：{task['title']} 完成" if result == "done"
                else f"已记录：{task['title']} 做错 · 已上调相关技能的权重")


def render_question(thread_id: str, task: dict, can_mark: bool, is_new: bool) -> None:
    status = task.get("status", "todo")
    icon, body, diff, done_col, wrong_col = st.columns([0.05, 0.55, 0.12, 0.14, 0.14], vertical_alignment="center")
    icon.markdown(STATUS_ICON.get(status, STATUS_ICON["todo"]))
    new_badge = " :blue-badge[新增]" if is_new else ""
    body.markdown(f"{question_link(task)}{new_badge}  \n:gray[{task.get('requested_skill') or ''}]")
    diff.markdown(f":{'orange' if task.get('difficulty') == 'hard' else 'gray'}-badge[{DIFF_LABEL.get(task.get('difficulty'), task.get('difficulty'))}]")
    if can_mark:
        if done_col.button("完成", icon=":material/check:", key=f"done-{task['id']}", help="标记为完成"):
            mark(thread_id, task, "done")
            st.rerun()
        if wrong_col.button("做错", icon=":material/close:", key=f"wrong-{task['id']}", help="标记为做错"):
            mark(thread_id, task, "wrong")
            st.rerun()


def render_days(thread_id: str, state: dict, today: int) -> None:
    days = state["days"]
    summaries = {s["day"]: s["summary"] for s in state.get("summaries") or []}
    new_ids = ss.new_ids.get(thread_id, set())
    for i, day in enumerate(days):
        finished = sum(t.get("status", "todo") != "todo" for t in day)
        if i < today:
            meta = f"{finished}/{len(day)} 已完成" + (f" · {len(day) - finished} 道未完成" if finished < len(day) else "")
        elif i == today:
            meta = f"今天 · 已完成 {finished}/{len(day)}"
        else:
            meta = f"{len(day)} 题"
        new_count = sum(t["id"] in new_ids for t in day)
        label = f"第 {i + 1} 天 · {meta}"
        if i == len(days) - 1 and i > today:
            label += " · 复习日"
        if new_count:
            label += f" · 新增 {new_count} 题"
        with st.expander(label, expanded=(i == today or new_count > 0), icon=":material/today:" if i == today else None):
            for task in day:
                can_mark = i <= today and task.get("status", "todo") == "todo"
                render_question(thread_id, task, can_mark, task["id"] in new_ids)
            if summaries.get(i + 1):
                st.caption(summaries[i + 1])


def render_message(message) -> None:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage) and message.tool_calls:
        # Show what the agent decided to do, so the ReAct steps are visible in the demo.
        for call in message.tool_calls:
            args = ", ".join(f"{k}={v}" for k, v in call["args"].items() if k != "reason")
            st.caption(f":material/build: {TOOL_LABEL.get(call['name'], call['name'])}（{call['name']}）{args}")
            if call["args"].get("reason"):
                st.caption(f"原因：{call['args']['reason']}")
    elif isinstance(message, ToolMessage):
        st.caption(f":material/subdirectory_arrow_right: {message.content}")
    elif isinstance(message, AIMessage) and message.content:
        with st.chat_message("assistant"):
            st.markdown(message.content)


def render_chat(thread_id: str, state: dict) -> None:
    st.subheader("对话")
    st.caption("告诉我做题进展，我会调整明天之后的计划")
    box = st.container(height=560)
    with box:
        messages = state.get("messages") or []
        if not messages:
            with st.chat_message("assistant"):
                st.markdown(WELCOME)
        for message in messages:
            render_message(message)

    prompt = st.chat_input("比如：今天第二题做错了 / 我实在不会 SQL 窗口函数")
    if not prompt:
        return
    before = {t["id"] for day in state["days"] for t in day}
    with box:
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("思考中……"):
            try:
                graph.chat(thread_id, prompt)
            except Exception as error:
                st.error(f"出错了：{error}")
                return
    after = {t["id"] for day in graph.get_state(thread_id)["days"] for t in day}
    ss.new_ids[thread_id] = after - before
    st.rerun()


def render_plan_page(thread_id: str) -> None:
    state = graph.get_state(thread_id)
    days = state.get("days") or []
    if not days:
        st.warning("这个会话没有已保存的计划。")
        return
    today = min(today_index(state), len(days) - 1)
    tasks = [t for day in days for t in day]
    finished = sum(t.get("status", "todo") != "todo" for t in tasks)

    head, progress = st.columns([3, 2], vertical_alignment="bottom")
    head.title(graph.plan_title(state))
    head.caption(f"{len(days)} 天计划 · 今天是第 {today + 1} 天 · 剩余 {len(days) - 1 - today} 天")
    progress.progress(finished / len(tasks) if tasks else 0.0, text=f"总进度 {finished}/{len(tasks)}")

    plan_col, chat_col = st.columns([3, 2], gap="large")
    with plan_col:
        render_days(thread_id, state, today)
    with chat_col:
        render_chat(thread_id, state)


render_sidebar()
if ss.thread_id:
    render_plan_page(ss.thread_id)
else:
    render_new_plan()
