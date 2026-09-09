import streamlit as st
from orchestration import multi_agent
import uuid

st.set_page_config(page_title="DS-Interview-Copilot", layout="centered")

if "chats" not in st.session_state:
    st.session_state.chats = []

if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

with st.sidebar:
    if st.button("New Chat", use_container_width=True):
        st.session_state.current_chat_id = None

    st.divider()
    for chat in st.session_state.chats:
        if st.button(chat["title"], key=chat["id"], use_container_width=True):
            st.session_state.current_chat_id = chat["id"]

def display_days(days, summaries):
    diff = {"easy": 1, "medium": 2, "hard": 3}
    summary_map = {s["day"]: s["summary"] for s in summaries} if summaries else {}

    for i, day in enumerate(days, 1):
        st.markdown(f"### Day {i}")
        st.caption(f"Difficulty score: {sum(diff[t['difficulty']] for t in day)}")

        for t in day:
            raw_title = t["title"]
            url = t.get("url")
            backup_url = t.get("backup_url")
            title_display = raw_title
            if t["category"] == "SQL":
                if backup_url:
                    title_display = f"[{raw_title}]({backup_url})"
            else:
                if url:
                    title_display = f"[{raw_title}]({url})"
                elif backup_url:
                    title_display = f"[{raw_title}]({backup_url})"
                difficulty_display = ""
            if t["category"] in ["SQL", "Algorithms","Pandas"]:
                difficulty_display = f" ({t['difficulty']})"
            
            st.markdown(
                f"- **{t['type']}_{t['category']}** - {title_display}"
                f"{difficulty_display}"
                )
        if i in summary_map:
            st.info(summary_map[i])

        st.divider()

st.markdown("# 🤖 DS Interview Copilot")
st.caption("Your AI copilot for data science technical interview preparation")

if st.session_state.current_chat_id is None:
    st.markdown("### Prepare for your next interview")

    jd = st.text_area(
        "Job Description",
        height=180,
        placeholder="Paste the job description here..."
    )

    user_desc = st.text_area(
        "User Profile",
        height=140,
        placeholder="Describe your background, skills, requirements..."
    )

    days_left = st.slider(
        "Days left before interview",
        min_value=1,
        max_value=15,
        value=7,
        step=1,
        help="How many days do you have to prepare?"
    )


    if st.button("Generate Study Plan", type="primary"):
        if not jd.strip() and not user_desc.strip():
            st.warning("Please fill in both fields.")
        else:
            with st.spinner("Running multi-agent pipeline..."):
                result = multi_agent(jd, user_desc, days_left)
                days = result["days"]
                summaries = result["summaries"]

            chat = {
                "id": str(uuid.uuid4()),
                "title": f"Chat {len(st.session_state.chats) + 1}",
                "jd": jd,
                "user_desc": user_desc,
                "days_left": days_left,
                "days": days,
                "summaries": summaries
            }

            st.session_state.chats.insert(0, chat)
            st.session_state.current_chat_id = chat["id"]
            st.rerun()
else:
    chat = next(c for c in st.session_state.chats if c["id"] == st.session_state.current_chat_id)
    st.markdown(f"### 📌 {chat['title']}")
    st.caption(f"⏰ Total Days: {chat.get('days_left', 'N/A')}")
    with st.expander("🧠 User Background", expanded=False):
        st.write(chat["user_desc"])
    with st.expander("📄 Job Description", expanded=False):
        st.write(chat["jd"])
    st.divider()
    display_days(chat["days"], chat["summaries"])