#!/usr/bin/env python3
"""
Convert NumPy-style docstrings to Google-style docstrings.

Usage:
    python convert_docstrings.py <file_or_directory>

Examples:
    python convert_docstrings.py nltools/models/ridge.py
    python convert_docstrings.py nltools/
"""

import re
import sys
from pathlib import Path


def convert_numpy_section_to_google(section_name, content, is_returns=False):
    """
    Convert a single NumPy-style section to Google style.

    Args:
        section_name (str): Section name (Parameters, Attributes, Returns, etc.)
        content (str): Content after the section header and dashes
        is_returns (bool): Whether this is a Returns section (different format)

    Returns:
        str: Converted Google-style section
    """
    # Map NumPy section names to Google names
    name_map = {
        "Parameters": "Args",
        "Returns": "Returns",
        "Attributes": "Attributes",
        "Raises": "Raises",
        "Yields": "Yields",
        "See Also": "See Also",
        "Notes": "Notes",
        "References": "References",
        "Examples": "Examples",
        "Warns": "Warnings",
    }

    google_name = name_map.get(section_name, section_name)

    if is_returns:
        # Returns section: "type\n    description" -> "type: description"
        lines = content.strip().split("\n")
        if not lines or not lines[0].strip():
            return f"{google_name}:\n"

        # Try to find type line and description
        result = f"{google_name}:\n"
        type_line = lines[0].strip()

        # Check if there's a description
        desc_lines = []
        for line in lines[1:]:
            if line.strip():
                # Remove leading whitespace but preserve relative indentation
                desc_lines.append(line.strip())

        if desc_lines:
            result += f"    {type_line}: {' '.join(desc_lines)}\n"
        else:
            result += f"    {type_line}\n"

        return result

    # Parse parameter/attribute entries: "name : type\n    description"
    # Using regex to match entries
    pattern = r"^(\w+_?)\s*:\s*(.+?)$"

    lines = content.split("\n")
    result = f"{google_name}:\n"

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is a parameter/attribute line
        match = re.match(pattern, line.strip())
        if match:
            param_name = match.group(1)
            param_type = match.group(2).strip()

            # Collect description lines (indented continuation)
            desc_lines = []
            i += 1
            while i < len(lines):
                if lines[i] and not lines[i][0].isspace():
                    # Next parameter or end of section
                    break
                if lines[i].strip():  # Skip empty lines within description
                    # Preserve relative indentation for nested content (lists, etc.)
                    stripped = lines[i].strip()
                    desc_lines.append(stripped)
                elif desc_lines:  # Preserve blank lines within description
                    desc_lines.append("")
                i += 1

            # Format as Google style: "    name (type): description"
            description = " ".join(desc_lines) if desc_lines else ""

            # Handle multi-line descriptions with lists/bullets
            if any(d.startswith(("-", "*", "•")) for d in desc_lines if d):
                # Has list items - format with proper indentation
                result += f"    {param_name} ({param_type}): "

                # First non-list-item text
                first_desc = []
                list_start_idx = None
                for idx, d in enumerate(desc_lines):
                    if d.startswith(("-", "*", "•")):
                        list_start_idx = idx
                        break
                    if d:
                        first_desc.append(d)

                if first_desc:
                    result += " ".join(first_desc) + "\n\n"
                else:
                    result += "\n\n"

                # Add list items with proper indentation
                if list_start_idx is not None:
                    for d in desc_lines[list_start_idx:]:
                        if d:
                            result += f"        {d}\n"
                        else:
                            result += "\n"
            else:
                # Simple single-line or flowing description
                result += f"    {param_name} ({param_type}): {description}\n"

            # Back up one line since the while loop will increment
            i -= 1

        i += 1

    return result


def convert_docstring(docstring):
    """
    Convert a NumPy-style docstring to Google style.

    Args:
        docstring (str): The docstring to convert

    Returns:
        str: Converted docstring
    """
    if not docstring:
        return docstring

    # Find NumPy-style sections
    # Pattern: Section name followed by dashes on next line
    section_pattern = r"^(\s*)(Parameters|Attributes|Returns|Raises|Yields|Warns|See Also|Notes|References|Examples)\s*\n\1-+\s*\n"

    # Split docstring into parts: before first section, sections, after last section
    parts = re.split(section_pattern, docstring, flags=re.MULTILINE)

    if len(parts) == 1:
        # No NumPy sections found
        return docstring

    # Reconstruct with converted sections
    result = parts[0]  # Content before first section

    i = 1
    while i < len(parts):
        if i + 2 < len(parts):
            indent = parts[i]
            section_name = parts[i + 1]

            # Find the content of this section (until next section or end)
            # Look ahead to find where this section ends
            section_content = parts[i + 2] if i + 2 < len(parts) else ""

            # Find next section
            next_section_match = re.search(
                r"^(\s*)(Parameters|Attributes|Returns|Raises|Yields|Warns|See Also|Notes|References|Examples)\s*\n\1-+",
                section_content,
                flags=re.MULTILINE,
            )

            if next_section_match:
                # Content ends where next section starts
                content = section_content[: next_section_match.start()]
                remaining = section_content[next_section_match.start() :]
            else:
                content = section_content
                remaining = ""

            # Convert this section
            is_returns = section_name == "Returns"
            converted = convert_numpy_section_to_google(
                section_name, content, is_returns
            )

            result += indent + converted

            # If there's remaining content, we need to parse it
            if remaining:
                # Re-add to parts for processing
                parts[i + 2] = remaining
            else:
                i += 3
        else:
            i += 1

    return result


def process_file(file_path, dry_run=False):
    """
    Process a single Python file, converting NumPy docstrings to Google style.

    Args:
        file_path (Path): Path to the Python file
        dry_run (bool): If True, don't write changes, just report

    Returns:
        bool: True if file was modified
    """
    content = file_path.read_text()

    # Find all docstrings (triple-quoted strings)
    # This regex matches both ''' and """ docstrings
    docstring_pattern = r'("""|\'\'\')(.*?)\1'

    modified = False

    def replace_docstring(match):
        nonlocal modified
        quote = match.group(1)
        docstring = match.group(2)

        converted = convert_docstring(docstring)

        if converted != docstring:
            modified = True
            return f"{quote}{converted}{quote}"
        return match.group(0)

    new_content = re.sub(docstring_pattern, replace_docstring, content, flags=re.DOTALL)

    if modified and not dry_run:
        file_path.write_text(new_content)
        print(f"✅ Converted: {file_path}")
    elif modified:
        print(f"📝 Would convert: {file_path}")
    else:
        print(f"⏭️  No NumPy docstrings: {file_path}")

    return modified


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")

    files_to_process = []

    if target.is_file():
        files_to_process = [target]
    elif target.is_dir():
        files_to_process = sorted(target.rglob("*.py"))
    else:
        print(f"Error: {target} not found")
        sys.exit(1)

    print(f"Processing {len(files_to_process)} Python files...\n")

    modified_count = 0
    for file_path in files_to_process:
        if process_file(file_path, dry_run):
            modified_count += 1

    print(
        f"\n📊 Summary: {modified_count}/{len(files_to_process)} files {'would be ' if dry_run else ''}modified"
    )


if __name__ == "__main__":
    main()
