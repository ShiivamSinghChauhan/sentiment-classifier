"""Dashboard for classified feedback.  Run: streamlit run app.py"""

import pandas as pd
import streamlit as st

import insights
import store

st.set_page_config(page_title="Feedback classifier", page_icon=":material/reviews:", layout="wide")
st.title("Customer feedback classifier")

with st.container(border=True):
    st.subheader("Classify feedback")
    text = st.text_area("One feedback per line", height=120, placeholder="The app crashes when I open my cart.")
    upload = st.file_uploader("Or upload a CSV with a 'feedback' column", type="csv")
    if st.button("Classify", type="primary", icon=":material/label:"):
        texts = text.splitlines()
        if upload is not None:
            texts += pd.read_csv(upload)["feedback"].astype(str).tolist()
        with st.spinner(f"Classifying {len(texts)} items..."):
            results = store.classify_and_save(texts)
        st.dataframe(pd.DataFrame(results).drop(columns=["confidence"]), hide_index=True,
                     column_config={"action_items": st.column_config.ListColumn("Action items")})

stats = store.stats()
if not stats["total"]:
    st.info("Nothing classified yet.")
    st.stop()

left, middle, right = st.columns(3)
left.metric("Feedback", stats["total"])
middle.metric("High urgency", stats["urgency"].get("high", 0))
right.metric("Sent to the LLM for a second opinion", stats["reviewed_by_llm"])

charts = st.columns(3)
for column, field in zip(charts, ["sentiment", "category", "urgency"]):
    with column:
        st.caption(field.capitalize())
        st.bar_chart(pd.Series(stats[field], name="count"), horizontal=True)

st.subheader("What to fix first")
st.caption("Priority adds up every item's urgency (high 3, medium 2, low 1) times its negativity "
           "(negative 1, neutral 0.3, positive 0), so it grows with volume, negativity and urgency.")
table = insights.priorities()
st.dataframe(table, hide_index=True, alt="Categories ranked by priority",
             column_config={
                 "negative_share": st.column_config.ProgressColumn("Negative", min_value=0, max_value=1, format="percent"),
                 "negative_last_7_days": st.column_config.NumberColumn("Negative, last 7 days"),
                 "negative_7_days_before": st.column_config.NumberColumn("Negative, 7 days before"),
                 "top_actions": st.column_config.ListColumn("Top action items"),
             })
st.download_button("Download as CSV", table.assign(top_actions=table["top_actions"].str.join("; ")).to_csv(index=False),
                   file_name="priorities.csv", mime="text/csv", icon=":material/download:")

rows = pd.DataFrame(store.all_feedback())
urgency = st.segmented_control("Urgency", ["high", "medium", "low"], default="high")
shown = rows if urgency is None else rows[rows["urgency"] == urgency]
st.dataframe(shown[["created_at", "text", "sentiment", "category", "urgency", "summary", "action_items"]],
             hide_index=True, column_config={"action_items": st.column_config.ListColumn("Action items")})

if st.button("Write a trend report", icon=":material/insights:"):
    with st.spinner("Reading the feedback..."):
        report = insights.report()
    st.subheader(report["headline"])
    for title in ("top_risks", "top_wins", "actions"):
        st.markdown(f"**{title.replace('_', ' ').capitalize()}**")
        st.markdown("\n".join(f"- {item}" for item in report.get(title, [])) or "-")
