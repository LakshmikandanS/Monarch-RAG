import re

def get_sections_with_hierarchy(content):
    # 1. Clean the content of windows style carriage returns just in case
    cleaned_content = content.replace('\r\n', '\n')
    
    # 2. Resilient pattern: Looks for 1 to 6 '#' signs at the start of any line
    # followed by at least one space or tab, capturing the text until the end of that line.
    header_pattern = re.compile(r'^(#{1,6})[ \t]+(.*)$', re.MULTILINE)
    
    sections = []
    hierarchy_stack = []

    for match in header_pattern.finditer(cleaned_content):
        # match.group(1) is the actual hashtags string (e.g. "##")
        level = len(match.group(1))        
        header_text = match.group(2).strip() 
        start_char = match.start()
        
        # Maintain the stack hierarchy
        while hierarchy_stack and hierarchy_stack[-1][0] >= level:
            hierarchy_stack.pop()
            
        hierarchy_stack.append((level, header_text))
        hierarchy_path = " > ".join([h[1] for h in hierarchy_stack])
        
        sections.append({
            "start_char": start_char,
            "section_path": hierarchy_path
        })
        
    return sections