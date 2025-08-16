"""
Advanced Data Processor - Working Implementation
Author: Arjun Loya
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataProcessor:
    """Advanced data processing with statistical analysis."""
    
    def __init__(self):
        self.analysis_results = {}
    
    def load_and_analyze_data(self, file_path):
        """Load and perform comprehensive analysis."""
        try:
            # Load data with encoding support
            if file_path.endswith('.csv'):
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                df = None
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        continue
                        
                if df is None:
                    raise ValueError("Could not read CSV with any supported encoding")
            else:
                df = pd.read_excel(file_path)
            
            # Perform comprehensive analysis
            results = self._analyze_dataframe(df)
            return results
            
        except Exception as e:
            raise Exception(f"Data processing error: {e}")
    
    def _analyze_dataframe(self, df):
        """Perform comprehensive DataFrame analysis."""
        results = {
            'metadata': self._get_metadata(df),
            'trends': self._analyze_trends(df),
            'insights': self._generate_insights(df),
            'business_domain': self._detect_domain(df),
            'kpis': self._calculate_kpis(df),
            'correlations': self._analyze_correlations(df)
        }
        return results
    
    def _get_metadata(self, df):
        """Extract metadata from DataFrame."""
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'data_quality_score': self._calculate_quality_score(df),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
    
    def _calculate_quality_score(self, df):
        """Calculate data quality score."""
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        uniqueness = 1 - (df.duplicated().sum() / len(df))
        quality_score = (completeness * 0.7 + uniqueness * 0.3) * 100
        return min(100, max(0, quality_score))
    
    def _analyze_trends(self, df):
        """Analyze trends in numeric columns."""
        trends = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Analyze top 5 columns
            if len(df[col].dropna()) > 10:
                trend_data = self._analyze_column_trend(df[col])
                trends[col] = trend_data
        
        return trends
    
    def _analyze_column_trend(self, series):
        """Analyze trend for a single column."""
        clean_series = series.dropna()
        
        if len(clean_series) < 3:
            return {'trend_direction': 'insufficient_data'}
        
        # Linear regression
        x = np.arange(len(clean_series))
        y = clean_series.values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determine trend
        if abs(slope) < np.std(y) * 0.1:
            direction = 'stable'
            strength = 'weak'
        elif slope > 0:
            direction = 'increasing'
            strength = 'strong' if r_value**2 > 0.7 else 'moderate' if r_value**2 > 0.3 else 'weak'
        else:
            direction = 'decreasing'
            strength = 'strong' if r_value**2 > 0.7 else 'moderate' if r_value**2 > 0.3 else 'weak'
        
        # Calculate percentage change
        first_val = y[0]
        last_val = y[-1]
        pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
        
        return {
            'trend_direction': direction,
            'trend_strength': strength,
            'total_change_percent': pct_change,
            'current_value': last_val,
            'average_value': np.mean(y),
            'r_squared': r_value**2,
            'slope': slope,
            'volatility_level': 'high' if np.std(y)/np.mean(y) > 0.3 else 'low'
        }
    
    def _detect_domain(self, df):
        """Detect business domain from column names."""
        columns_text = ' '.join(df.columns).lower()
        
        domain_keywords = {
            'sales': ['revenue', 'sales', 'orders', 'customers', 'deals', 'units'],
            'marketing': ['campaign', 'impressions', 'clicks', 'ctr', 'roas', 'leads'],
            'finance': ['profit', 'margin', 'expenses', 'cost', 'budget', 'cash'],
            'hr': ['employee', 'salary', 'performance', 'training'],
            'operations': ['efficiency', 'productivity', 'capacity', 'inventory']
        }
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in columns_text)
            domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else 'general'
    
    def _calculate_kpis(self, df, domain=None):
        """Calculate relevant KPIs."""
        kpis = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            primary_col = numeric_cols[0]
            kpis[f'{primary_col}_total'] = df[primary_col].sum()
            kpis[f'{primary_col}_average'] = df[primary_col].mean()
            kpis['total_records'] = len(df)
            
            if len(numeric_cols) > 1:
                secondary_col = numeric_cols[1] 
                kpis['efficiency_ratio'] = df[primary_col].sum() / df[secondary_col].sum() if df[secondary_col].sum() != 0 else 0
        
        return kpis
    
    def _analyze_correlations(self, df):
        """Analyze correlations between numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        return {
            'matrix': corr_matrix.to_dict(),
            'summary': {
                'avg_correlation': np.mean(np.abs(corr_matrix.values)),
                'max_correlation': np.max(corr_matrix.values),
                'min_correlation': np.min(corr_matrix.values)
            }
        }
    
    def _generate_insights(self, df):
        """Generate business insights."""
        insights = []
        
        # Basic insights
        insights.append(f"Dataset contains {len(df):,} records across {len(df.columns)} dimensions")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            primary_metric = numeric_cols[0]
            total_value = df[primary_metric].sum()
            insights.append(f"Total {primary_metric.replace('_', ' ')}: ${total_value:,.0f}")
        
        # Data quality insight
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        if completeness > 0.95:
            insights.append("Excellent data quality enables confident analysis")
        elif completeness > 0.8:
            insights.append("Good data quality with minor gaps")
        else:
            insights.append("Data quality improvements recommended")
        
        return insights