"""
Export Utilities Module
Exports data and reports in various formats (PDF, Excel, JSON)
"""

import pandas as pd
import json
from datetime import datetime
import os
from io import BytesIO

class ExportManager:
    """Handles exporting data and reports"""
    
    def __init__(self):
        pass
    
    def export_to_excel(self, data_dict, filename=None):
        """Export multiple dataframes to Excel"""
        if filename is None:
            filename = f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                elif isinstance(df, dict):
                    pd.DataFrame([df]).to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
        return filename
    
    def export_to_csv(self, df, filename=None):
        """Export dataframe to CSV"""
        if filename is None:
            filename = f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        df.to_csv(filename, index=False)
        return filename
    
    def export_to_json(self, data, filename=None):
        """Export data to JSON"""
        if filename is None:
            filename = f'export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Handle pandas dataframes
        if isinstance(data, pd.DataFrame):
            data = data.to_dict('records')
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        return filename
    
    def generate_report_json(self, analysis_results, model_info, statistics):
        """Generate comprehensive report in JSON format"""
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'sentiment_analysis_report'
            },
            'model_information': model_info,
            'statistics': statistics,
            'analysis_results': analysis_results
        }
        
        filename = f'report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        self.export_to_json(report, filename)
        return filename
    
    def create_summary_report(self, df, model_info, output_dir='reports'):
        """Create a comprehensive summary report"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare data for Excel export
        excel_data = {
            'Summary': pd.DataFrame([{
                'Total Reviews': len(df),
                'Model Accuracy': model_info.get('accuracy', 'N/A'),
                'Model Type': model_info.get('model_type', 'N/A'),
                'Report Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }]),
            'Sentiment Distribution': df['Sentiment'].value_counts().reset_index() if 'Sentiment' in df.columns else pd.DataFrame(),
            'Category Distribution': df['Category'].value_counts().reset_index() if 'Category' in df.columns else pd.DataFrame(),
            'Sample Data': df.head(100)
        }
        
        # Export to Excel
        excel_file = os.path.join(output_dir, f'summary_report_{timestamp}.xlsx')
        self.export_to_excel(excel_data, excel_file)
        
        # Export to JSON
        json_data = {
            'summary': {
                'total_reviews': len(df),
                'model_info': model_info
            },
            'sentiment_distribution': df['Sentiment'].value_counts().to_dict() if 'Sentiment' in df.columns else {},
            'category_distribution': df['Category'].value_counts().to_dict() if 'Category' in df.columns else {}
        }
        
        json_file = os.path.join(output_dir, f'summary_report_{timestamp}.json')
        self.export_to_json(json_data, json_file)
        
        return {
            'excel_file': excel_file,
            'json_file': json_file
        }
    
    def export_predictions_batch(self, predictions, format='json'):
        """Export batch predictions"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == 'json':
            filename = f'predictions_{timestamp}.json'
            return self.export_to_json(predictions, filename)
        elif format == 'csv':
            df = pd.DataFrame(predictions)
            filename = f'predictions_{timestamp}.csv'
            return self.export_to_csv(df, filename)
        elif format == 'excel':
            df = pd.DataFrame(predictions)
            filename = f'predictions_{timestamp}.xlsx'
            return self.export_to_excel({'Predictions': df}, filename)
        else:
            raise ValueError(f"Unsupported format: {format}")

