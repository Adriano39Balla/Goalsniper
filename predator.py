from model_manager import load_model
from utils import logger

def generate_prediction(model, label_encoder, home_stats, away_stats):
    features = [[
        home_stats['avg_scored'],
        home_stats['avg_conceded'],
        away_stats['avg_scored'],
        away_stats['avg_conceded']
    ]]
    pred_enc = model.predict(features)[0]
    pred_prob = model.predict_proba(features)[0]
    pred_label = label_encoder.inverse_transform([pred_enc])[0]
    prob_dict = dict(zip(label_encoder.classes_, pred_prob))
    return pred_label, prob_dict

def format_predictions(predictions):
    outcome_map = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
    over_under_map = {'Over': 'Over 2.5 Goals', 'Under': 'Under 2.5 Goals'}
    btts_map = {'Yes': 'Both Teams To Score', 'No': 'No BTTS'}

    outcome_label, outcome_prob = predictions['outcome']
    ou_label, ou_prob = predictions['over_under']
    btts_label, btts_prob = predictions['btts']

    msg = (
        f"Match Result: {outcome_map.get(outcome_label, 'Unknown')} "
        f"(H: {outcome_prob.get('H',0):.2f}, D: {outcome_prob.get('D',0):.2f}, A: {outcome_prob.get('A',0):.2f})\n"
        f"Over/Under 2.5 Goals: {over_under_map.get(ou_label, 'Unknown')} "
        f"(Over: {ou_prob.get('Over',0):.2f}, Under: {ou_prob.get('Under',0):.2f})\n"
        f"BTTS: {btts_map.get(btts_label, 'Unknown')} "
        f"(Yes: {btts_prob.get('Yes',0):.2f}, No: {btts_prob.get('No',0):.2f})"
    )
    return msg
