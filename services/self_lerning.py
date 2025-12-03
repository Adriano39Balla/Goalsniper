import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split

from utils.database import DatabaseManager
from utils.logger import logger
from models.random_forest import BettingPredictor
from utils.feature_engineering import FeatureEngineer

class SelfLearningSystem:
    """Self-learning system that improves from incorrect predictions"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.predictor = BettingPredictor()
        self.feature_engineer = FeatureEngineer()
        
    def analyze_results(self, days_back: int = 7):
        """Analyze recent bet results for learning"""
        
        logger.info(f"Analyzing results from last {days_back} days...")
        
        # Get recent results
        query = """
        SELECT p.*, br.*, p.features as original_features
        FROM predictions p
        JOIN bet_results br ON p.id = br.prediction_id
        WHERE br.analyzed_at >= NOW() - INTERVAL '%s days'
        AND p.model_version = %s
        ORDER BY br.analyzed_at DESC
        """
        
        results = self.db.execute_query(
            query, 
            (days_back, os.getenv('MODEL_VERSION', 'v1.0')), 
            fetch=True
        )
        
        if not results:
            logger.warning("No recent results found for analysis")
            return
        
        # Group by prediction type
        results_by_type = {}
        for result in results:
            pred_type = result['prediction_type']
            if pred_type not in results_by_type:
                results_by_type[pred_type] = []
            results_by_type[pred_type].append(result)
        
        # Analyze each type
        insights = {}
        for pred_type, type_results in results_by_type.items():
            insights[pred_type] = self.analyze_prediction_type(type_results)
        
        # Generate learning recommendations
        recommendations = self.generate_recommendations(insights)
        
        # Apply improvements if needed
        if recommendations.get('retrain', False):
            self.retrain_models(insights)
        
        logger.info("Self-learning analysis completed")
        return insights, recommendations
    
    def analyze_prediction_type(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze results for a specific prediction type"""
        
        total = len(results)
        correct = sum(1 for r in results if r['is_correct'])
        incorrect = total - correct
        
        accuracy = correct / total if total > 0 else 0
        
        # Analyze patterns in incorrect predictions
        false_positives = []
        false_negatives = []
        
        for result in results:
            if not result['is_correct']:
                # Analyze features that might have caused error
                error_analysis = self.analyze_error(result)
                
                if error_analysis.get('type') == 'false_positive':
                    false_positives.append(error_analysis)
                else:
                    false_negatives.append(error_analysis)
        
        # Calculate confidence calibration
        confidence_scores = [r['confidence'] for r in results]
        correct_confidences = [r['confidence'] for r in results if r['is_correct']]
        incorrect_confidences = [r['confidence'] for r in results if not r['is_correct']]
        
        analysis = {
            'total_predictions': total,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'avg_correct_confidence': np.mean(correct_confidences) if correct_confidences else 0,
            'avg_incorrect_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
            'false_positives': len(false_positives),
            'false_negatives': len(false_negatives),
            'common_error_patterns': self.find_common_patterns(false_positives + false_negatives),
            'suggested_threshold_adjustment': self.calculate_threshold_adjustment(
                correct_confidences, incorrect_confidences
            )
        }
        
        return analysis
    
    def analyze_error(self, result: Dict) -> Dict[str, Any]:
        """Analyze individual prediction error"""
        
        features = result.get('original_features', {})
        prediction = result.get('prediction_value')
        actual = result.get('actual_result')
        confidence = result.get('confidence', 0)
        
        # Determine error type
        error_type = 'false_positive' if prediction != actual else 'false_negative'
        
        # Identify potential feature issues
        problematic_features = []
        
        # Check if confidence was too high for wrong prediction
        if confidence > 0.7 and not result['is_correct']:
            problematic_features.append('overconfidence')
        
        # Analyze specific features based on prediction type
        pred_type = result.get('prediction_type')
        
        if pred_type == 'over_under':
            if 'shots_on_goal' in features and features['shots_on_goal'] < 3:
                problematic_features.append('low_shooting_accuracy')
        
        return {
            'type': error_type,
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'problematic_features': problematic_features,
            'suggested_action': self.suggest_correction(error_type, pred_type)
        }
    
    def find_common_patterns(self, errors: List[Dict]) -> List[Dict]:
        """Find common patterns in errors"""
        
        if not errors:
            return []
        
        # Group errors by type and problematic features
        patterns = {}
        
        for error in errors:
            key = (error['type'], tuple(error.get('problematic_features', [])))
            
            if key not in patterns:
                patterns[key] = {
                    'error_type': error['type'],
                    'problematic_features': error.get('problematic_features', []),
                    'count': 0,
                    'avg_confidence': 0,
                    'examples': []
                }
            
            patterns[key]['count'] += 1
            patterns[key]['avg_confidence'] += error['confidence']
            
            if len(patterns[key]['examples']) < 3:
                patterns[key]['examples'].append({
                    'prediction': error['prediction'],
                    'actual': error['actual']
                })
        
        # Calculate averages and format
        result = []
        for pattern in patterns.values():
            pattern['avg_confidence'] /= pattern['count']
            pattern['frequency'] = pattern['count'] / len(errors)
            result.append(pattern)
        
        # Sort by frequency
        return sorted(result, key=lambda x: x['frequency'], reverse=True)
    
    def calculate_threshold_adjustment(self, correct_confidences: List[float],
                                     incorrect_confidences: List[float]) -> float:
        """Calculate suggested threshold adjustment"""
        
        if not correct_confidences or not incorrect_confidences:
            return 0
        
        avg_correct = np.mean(correct_confidences)
        avg_incorrect = np.mean(incorrect_confidences)
        
        # If incorrect predictions have high confidence, increase threshold
        if avg_incorrect > avg_correct:
            adjustment = (avg_incorrect - avg_correct) / 2
            return min(adjustment, 0.1)  # Max 10% adjustment
        
        return 0
    
    def suggest_correction(self, error_type: str, pred_type: str) -> str:
        """Suggest correction based on error type"""
        
        suggestions = {
            'false_positive': {
                '1X2': 'Increase required confidence threshold',
                'over_under': 'Add more weight to defensive statistics',
                'btts': 'Consider team defensive form more heavily'
            },
            'false_negative': {
                '1X2': 'Include more momentum-based features',
                'over_under': 'Add attacking intensity metrics',
                'btts': 'Consider recent scoring patterns'
            }
        }
        
        return suggestions.get(error_type, {}).get(pred_type, 'Review feature importance')
    
    def generate_recommendations(self, insights: Dict[str, Dict]) -> Dict[str, Any]:
        """Generate learning recommendations from insights"""
        
        recommendations = {
            'retrain': False,
            'adjustments': [],
            'new_features': [],
            'threshold_changes': {}
        }
        
        for pred_type, insight in insights.items():
            accuracy = insight.get('accuracy', 0)
            
            # Check if retraining is needed
            if accuracy < 0.6:  # Accuracy threshold
                recommendations['retrain'] = True
                recommendations['adjustments'].append(
                    f"{pred_type}: Low accuracy ({accuracy:.2%}), needs retraining"
                )
            
            # Check for threshold adjustments
            threshold_adj = insight.get('suggested_threshold_adjustment', 0)
            if abs(threshold_adj) > 0.02:  # Significant adjustment
                recommendations['threshold_changes'][pred_type] = threshold_adj
            
            # Check for common error patterns
            for pattern in insight.get('common_error_patterns', []):
                if pattern['frequency'] > 0.3:  # Frequent pattern
                    recommendations['new_features'].append(
                        f"{pred_type}: Add features to address {pattern['error_type']} "
                        f"(frequency: {pattern['frequency']:.1%})"
                    )
        
        return recommendations
    
    def retrain_models(self, insights: Dict[str, Dict]):
        """Retrain models based on insights"""
        
        logger.info("Starting model retraining based on insights...")
        
        # Get updated training data including recent results
        training_data = self.db.get_training_data(limit=20000)
        
        if len(training_data) < 1000:
            logger.warning("Insufficient training data for retraining")
            return
        
        # Prepare features and labels
        X, y = self.prepare_retraining_data(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Apply insights to modify training
        modified_trainer = self.apply_insights_to_training(insights)
        
        # Retrain models
        results = modified_trainer.train_all(
            X_train, y_train, X_test, y_test
        )
        
        # Log new performance
        for model_type, metrics in results.items():
            logger.info(f"Retrained {model_type} - New metrics: {metrics}")
        
        # Save new models
        self.predictor = modified_trainer
        
        logger.info("Model retraining completed successfully")
    
    def prepare_retraining_data(self, data: List[Dict]):
        """Prepare data for retraining"""
        
        # This would process the data into proper feature/label format
        # Implementation depends on your data structure
        pass
    
    def apply_insights_to_training(self, insights: Dict[str, Dict]):
        """Apply insights to modify training process"""
        
        # Create modified trainer based on insights
        modified_trainer = BettingPredictor()
        
        for pred_type, insight in insights.items():
            # Apply threshold adjustments
            threshold_adj = insight.get('suggested_threshold_adjustment', 0)
            
            # Adjust model parameters based on insights
            # This would modify the training process
            pass
        
        return modified_trainer
