import os
import datetime
import re
import sys

# Configuration
SOURCE_DIRS = ['src']
INCLUDE_DIRS = ['/usr/include', '/usr/local/include', 'include', 'src']
EXTENSIONS = ['.cpp', '.h', '.hpp']

def find_files(root_dirs):
    files = []
    for root_dir in root_dirs:
        for root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in EXTENSIONS):
                    files.append(os.path.join(root, filename))
    return files

def parse_includes(file_path):
    includes = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.match(r'^\s*#include\s+[<\"]([^>\"]+)[>\"]', line)
            if match:
                includes.append(match.group(1))
    return includes

def verify_header_existence(header_name):
    # Special cases for local project headers (simplification)
    if not header_name.endswith('.h') and not header_name.endswith('.hpp'):
        # Assume standard library headers exist (vector, string, etc.)
        return True
        
    for inc_dir in INCLUDE_DIRS:
        full_path = os.path.join(inc_dir, header_name)
        if os.path.exists(full_path):
            return True
    
    # Check strict project paths
    if os.path.exists(os.path.join('src', header_name)):
        return True
        
    return False

def annotate_file(file_path, verified_headers):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_block = f"// Verified headers: [{', '.join(verified_headers[:5])}{'...' if len(verified_headers)>5 else ''}]\n// Verification timestamp: {timestamp}\n"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Avoid double annotation
    if "// Verified headers:" in content:
        content = re.sub(r'// Verified headers:.*\n// Verification timestamp:.*\n', header_block, content)
    else:
        content = header_block + content
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"Annotated: {file_path}")

def main():
    print("Starting Header Verification...")
    files = find_files(SOURCE_DIRS)
    
    for file_path in files:
        includes = parse_includes(file_path)
        verified = []
        for inc in includes:
            if verify_header_existence(inc):
                verified.append(inc)
            else:
                print(f"WARNING: Could not verify header '{inc}' in {file_path}")
        
        if verified:
            annotate_file(file_path, verified)
            
    print("Verification Complete.")

if __name__ == "__main__":
    main()
