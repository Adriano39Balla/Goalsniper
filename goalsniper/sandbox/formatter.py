def format_tip_html(m):
    return (
        f"⚽ <b>{m['league']}</b>\n"
        f"{m['home_team']} vs {m['away_team']}\n"
        f"🕒 Kickoff: {m['kickoff_time']}\n\n"
        f"💡 Prediction: <b>{m['prediction']}</b>\n"
        f"📊 Confidence: {m['confidence']}\n\n"
        f"💰 Odds:\n"
        f" - Home Win: {m['odds']['home_win']}\n"
        f" - Draw: {m['odds']['draw']}\n"
        f" - Away Win: {m['odds']['away_win']}\n\n"
        f"📝 Note: {m['note']}"
    )
