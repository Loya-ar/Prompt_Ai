"""
Dynamic Prompt Engine - Real Prompt Engineering Implementation
Author: Arjun Loya
"""

class DynamicPromptEngine:
    """Advanced prompt engineering with context adaptation."""
    
    def __init__(self):
        self.style_templates = {
            'executive_summary': {
                'tone': 'formal, strategic, high-level',
                'focus': 'key insights, strategic implications',
                'length': 'concise but comprehensive'
            },
            'detailed_analysis': {
                'tone': 'analytical, thorough, data-driven',
                'focus': 'detailed findings, statistical significance',
                'length': 'comprehensive with supporting data'
            },
            'board_presentation': {
                'tone': 'formal, confident, results-oriented',
                'focus': 'bottom-line impact, strategic direction',
                'length': 'structured for presentation delivery'
            },
            'strategic_planning': {
                'tone': 'forward-looking, actionable, strategic',
                'focus': 'opportunities, recommendations, next steps',
                'length': 'detailed with action items'
            }
        }
        
        self.audience_adaptations = {
            'executive': 'Focus on strategic implications and business impact',
            'manager': 'Include operational details and actionable insights',
            'analyst': 'Provide technical depth and statistical analysis',
            'board_of_directors': 'Emphasize governance and strategic oversight'
        }
        
        self.domain_contexts = {
            'sales': 'revenue optimization, customer acquisition, market performance',
            'marketing': 'campaign effectiveness, ROI, customer engagement',
            'finance': 'profitability, cost management, financial health',
            'operations': 'efficiency, productivity, operational excellence',
            'general': 'overall business performance and strategic insights'
        }
    
    def generate_dynamic_prompt(self, analysis_results, narrative_style, audience):
        """Generate context-aware prompts based on analysis results."""
        
        # Extract analysis context
        metadata = analysis_results.get('metadata', {})
        trends = analysis_results.get('trends', {})
        business_domain = analysis_results.get('business_domain', 'general')
        insights = analysis_results.get('insights', [])
        kpis = analysis_results.get('kpis', {})
        
        # Get style and audience context
        style_context = self.style_templates.get(narrative_style, self.style_templates['executive_summary'])
        audience_context = self.audience_adaptations.get(audience, 'Provide professional business insights')
        domain_context = self.domain_contexts.get(business_domain, 'general business performance')
        
        # Build dynamic prompt
        prompt = f"""You are a senior business intelligence analyst and executive communication specialist. 

ANALYSIS CONTEXT:
- Dataset: {metadata.get('total_rows', 0):,} records across {metadata.get('total_columns', 0)} dimensions
- Business Domain: {business_domain.title()}
- Data Quality: {metadata.get('data_quality_score', 0):.1f}/100
- Analysis Focus: {domain_context}

KEY PERFORMANCE TRENDS:
"""
        
        # Add trend analysis to prompt
        if trends:
            for metric, trend_data in list(trends.items())[:3]:
                direction = trend_data.get('trend_direction', 'stable')
                change_pct = trend_data.get('total_change_percent', 0)
                current_val = trend_data.get('current_value', 0)
                strength = trend_data.get('trend_strength', 'weak')
                
                prompt += f"- {metric.replace('_', ' ').title()}: {direction} trend ({change_pct:+.1f}%), {strength} correlation, current value ${current_val:,.0f}\n"
        
        # Add insights to prompt
        if insights:
            prompt += f"\nKEY INSIGHTS:\n"
            for i, insight in enumerate(insights[:3], 1):
                prompt += f"{i}. {insight}\n"
        
        # Add KPIs to prompt
        if kpis:
            prompt += f"\nKEY PERFORMANCE INDICATORS:\n"
            for kpi_name, kpi_value in list(kpis.items())[:3]:
                if isinstance(kpi_value, (int, float)):
                    prompt += f"- {kpi_name.replace('_', ' ').title()}: {kpi_value:,.2f}\n"
        
        # Add style and audience instructions
        prompt += f"""
REPORT REQUIREMENTS:
- Style: {narrative_style.replace('_', ' ').title()} - {style_context['tone']}
- Audience: {audience.title()} - {audience_context}
- Focus Areas: {style_context['focus']}
- Length: {style_context['length']}

INSTRUCTIONS:
1. Generate a professional business intelligence report
2. Start with an executive summary highlighting key findings
3. Include specific metrics and data points from the analysis
4. Provide actionable strategic recommendations
5. Use professional business language appropriate for {audience} audience
6. Focus on {domain_context}
7. Maintain {style_context['tone']} throughout
8. Structure the report with clear sections and bullet points
9. Include quantitative evidence to support all claims
10. End with clear next steps and strategic priorities

Generate a comprehensive {narrative_style.replace('_', ' ')} report now:"""
        
        return prompt
    
    def generate_follow_up_prompt(self, initial_response, feedback_type):
        """Generate follow-up prompts for refinement."""
        
        follow_up_prompts = {
            'more_detail': "Please expand on the strategic recommendations section with more specific actionable steps and timeline considerations.",
            'executive_focus': "Please rewrite this with more focus on executive-level strategic implications and less operational detail.",
            'add_risks': "Please add a risk analysis section identifying potential challenges and mitigation strategies.",
            'add_opportunities': "Please enhance the opportunities section with market analysis and growth potential.",
            'financial_focus': "Please add more financial analysis including ROI, cost-benefit analysis, and budget implications."
        }
        
        return follow_up_prompts.get(feedback_type, "Please refine the analysis with additional insights.")
    
    def validate_prompt_quality(self, prompt):
        """Validate prompt quality and completeness."""
        quality_indicators = {
            'has_context': bool(any(word in prompt.lower() for word in ['analysis', 'data', 'trends'])),
            'has_instructions': bool('INSTRUCTIONS:' in prompt),
            'has_metrics': bool(any(word in prompt.lower() for word in ['$', '%', 'records'])),
            'has_audience': bool(any(word in prompt.lower() for word in ['executive', 'manager', 'analyst'])),
            'proper_length': len(prompt.split()) > 100
        }
        
        quality_score = sum(quality_indicators.values()) / len(quality_indicators) * 100
        
        return {
            'quality_score': quality_score,
            'indicators': quality_indicators,
            'recommendations': self._get_prompt_recommendations(quality_indicators)
        }
    
    def _get_prompt_recommendations(self, indicators):
        """Get recommendations for prompt improvement."""
        recommendations = []
        
        if not indicators['has_context']:
            recommendations.append("Add more data analysis context")
        if not indicators['has_instructions']:
            recommendations.append("Include clear generation instructions")
        if not indicators['has_metrics']:
            recommendations.append("Include specific metrics and numbers")
        if not indicators['proper_length']:
            recommendations.append("Expand prompt with more detail")
            
        return recommendations if recommendations else ["Prompt quality is excellent"]