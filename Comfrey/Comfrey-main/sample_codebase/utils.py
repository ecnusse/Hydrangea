
def segment_text(text, max_length=100):
    """Segment text into chunks"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) > max_length:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
        
        current_chunk.append(word)
        current_length += len(word) + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def validate_output(output, expected_format):
    """Validate LLM output against expected format"""
    if expected_format == "json":
        try:
            json.loads(output)
            return True
        except:
            return False
    elif expected_format == "code":
        try:
            compile(output, "<string>", "exec")
            return True
        except:
            return False
    return True
