REASONING_TASKS = {
    'chain_of_thought': ['problem_solving', 'math', 'logic', 'analysis'],
    'meta_reasoning': ['planning', 'strategy', 'evaluation', 'optimization'],
    'pattern_recognition': ['sequences', 'shapes', 'relationships', 'trends'],
    'react': ['decision_making', 'emergency', 'social_situations', 'technical_issues'],
    'reflection': ['learning', 'experience', 'improvement', 'self_analysis'],
    'toolformer': ['tool_selection', 'problem_solving', 'efficiency', 'automation']
}

REASONING_TOPICS = {
    'problem_solving': ['analytical', 'creative', 'critical', 'systematic'],
    'math': ['arithmetic', 'algebra', 'geometry', 'statistics'],
    'logic': ['deduction', 'induction', 'reasoning', 'inference'],
    'analysis': ['data', 'process', 'system', 'behavior'],
    'planning': ['strategy', 'scheduling', 'resource_allocation', 'risk_management'],
    'strategy': ['goals', 'methods', 'implementation', 'evaluation'],
    'evaluation': ['assessment', 'measurement', 'comparison', 'benchmarking'],
    'optimization': ['improvement', 'efficiency', 'performance', 'refinement'],
    'sequences': ['patterns', 'series', 'progressions', 'cycles'],
    'shapes': ['geometry', 'spatial', 'visual', 'structural'],
    'relationships': ['correlations', 'dependencies', 'connections', 'associations'],
    'trends': ['patterns', 'movements', 'changes', 'developments'],
    'decision_making': ['choices', 'consequences', 'tradeoffs', 'priorities'],
    'emergency': ['crisis', 'response', 'management', 'resolution'],
    'social_situations': ['interaction', 'communication', 'behavior', 'etiquette'],
    'technical_issues': ['problems', 'solutions', 'troubleshooting', 'maintenance'],
    'learning': ['education', 'training', 'development', 'growth'],
    'experience': ['practice', 'observation', 'application', 'insight'],
    'improvement': ['enhancement', 'development', 'progress', 'advancement'],
    'self_analysis': ['reflection', 'awareness', 'understanding', 'evaluation'],
    'tool_selection': ['choice', 'appropriateness', 'effectiveness', 'efficiency'],
    'automation': ['process', 'workflow', 'system', 'integration']
}

OUTPUT_DIR = 'Script/Generate/REASONING_OUTPUT'
MAX_RETRIES = 3
RETRY_DELAY = 2
NUM_SAMPLES_PER_TASK = 100
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'