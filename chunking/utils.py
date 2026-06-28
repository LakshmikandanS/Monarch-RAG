import re
from typing import Dict, List

def get_sections_with_hierarchy(content: str) -> List[Dict]:
    """
    Extracts Markdown headers and builds a hierarchical path (e.g., 'Chapter 1 > Training Loop').
    """
    # Clean Windows-style carriage returns
    cleaned_content = content.replace('\r\n', '\n')
    
    # Resilient pattern: Looks for 1 to 6 '#' signs at the start of any line
    header_pattern = re.compile(r'^(#{1,6})[ \t]+(.*)$', re.MULTILINE)
    
    sections = []
    hierarchy_stack = []

    for match in header_pattern.finditer(cleaned_content):
        level = len(match.group(1))        
        header_text = match.group(2).strip() 
        start_char = match.start()
        
        # Maintain the stack hierarchy (pop headers of equal or lower importance)
        while hierarchy_stack and hierarchy_stack[-1][0] >= level:
            hierarchy_stack.pop()
            
        hierarchy_stack.append((level, header_text))
        hierarchy_path = " > ".join([h[1] for h in hierarchy_stack])
        
        sections.append({
            "start_char": start_char,
            "section_path": hierarchy_path,
            "level": level,
            "header": header_text
        })
        
    return sections