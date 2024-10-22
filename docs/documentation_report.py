import ast
import os
from collections import defaultdict

from bs4 import BeautifulSoup


class ParentNodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.parent = None

    def visit(self, node):
        previous_parent = self.parent
        self.parent = node
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self.visit(child)
        self.parent = previous_parent


class DocAnalyzer:
    def __init__(self, source_dir, docs_dir):
        self.source_dir = source_dir
        self.docs_dir = docs_dir
        self.source_items = defaultdict(dict)
        self.doc_items = set()

    def analyze_source(self):
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith(".py"):
                    module_name = os.path.splitext(file)[0]
                    with open(os.path.join(root, file)) as f:
                        tree = ast.parse(f.read())
                        visitor = ParentNodeVisitor()
                        visitor.visit(tree)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                                item_name = node.name
                                parent_class = None
                                if (
                                    isinstance(node, ast.FunctionDef)
                                    and hasattr(node, "parent")
                                    and isinstance(node.parent, ast.ClassDef)
                                ):
                                    parent_class = node.parent.name
                                    item_name = f"{parent_class}.{node.name}"
                                full_name = f"{module_name}.{item_name}"
                                self.source_items[module_name][full_name] = {
                                    "type": (
                                        "function"
                                        if isinstance(node, ast.FunctionDef)
                                        else "class"
                                    ),
                                    "internal": node.name.startswith("_"),
                                    "has_docstring": ast.get_docstring(node)
                                    is not None,
                                    "parent_class": parent_class,
                                }

    def analyze_docs(self):
        for root, _, files in os.walk(self.docs_dir):
            for file in files:
                if file.endswith(".html"):
                    with open(os.path.join(root, file)) as f:
                        soup = BeautifulSoup(f, "html.parser")

                        # Find all class definitions
                        for class_def in soup.find_all("dl", class_="py class"):
                            class_name = class_def.find(
                                "dt", class_="sig sig-object py"
                            )["id"]
                            self.doc_items.add(class_name)

                            # Find all method definitions within the class
                            for method_def in class_def.find_all(
                                "dl", class_="py method"
                            ):
                                method_name = method_def.find(
                                    "dt", class_="sig sig-object py"
                                )["id"]
                                self.doc_items.add(method_name)

                        # Find all function definitions
                        for func_def in soup.find_all("dl", class_="py function"):
                            func_name = func_def.find("dt", class_="sig sig-object py")[
                                "id"
                            ]
                            self.doc_items.add(func_name)

                        # Find all property definitions
                        for prop_def in soup.find_all("dl", class_="py property"):
                            prop_name = prop_def.find("dt", class_="sig sig-object py")[
                                "id"
                            ]
                            self.doc_items.add(prop_name)

    def generate_html_report(self):
        html = """
        <html>
        <head>
            <style>
                body { font-family: Arial, sans-serif; }
                table { border-collapse: collapse; width: 90%; margin-bottom: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .highlight { background-color: #ffcccc; }
                .highlight-low { background-color: #ffebcc; }
                .tick { color: green; }
                .cross { color: red; }
            </style>
        </head>
        <body>
        <h1>Documentation Report</h1>
        <h3>Classes and methods that start with a single underscore are
          excluded from the report</h3>
        <h3>Red rows have a docstring, but are not in Sphinx, yellow have neither</h3>
        """

        for module, items in self.source_items.items():
            html += f"<details open><summary>Module: {module}</summary>"

            # Separate classes and functions
            classes = defaultdict(dict)
            functions = {}
            for full_name, info in items.items():
                if info["type"] == "class":
                    classes[full_name] = {"class_info": info, "methods": {}}
                elif info["parent_class"] is None:
                    functions[full_name] = info
                else:
                    parent_class_name = f"{module}.{info['parent_class']}"
                    classes[parent_class_name]["methods"][full_name] = info

            # Generate table for isolated functions
            html += self.generate_table("Isolated Functions", functions)

            # Generate table for each class
            for class_name, class_data in classes.items():
                class_items = {"class": {class_name: class_data["class_info"]}}
                class_items.update({"methods": class_data["methods"]})
                if class_name[0] == "_":
                    continue
                html += self.generate_table(
                    f'Class: {class_name.split(".")[-1]}', class_items
                )
            html += "</details>"
        html += "</body></html>"
        return html

    def generate_table(self, title, items):
        html = f"<h3>{title}</h3>"
        html += """
        <table>
            <tr>
                <th>Name</th>
                <th>Type</th>
                <th>Has Docstring</th>
                <th>In Sphinx</th>
            </tr>
        """

        if "class" in items:
            # This is a class table
            for name, info in items["class"].items():
                html += self.generate_table_row(name, info)
            for name, info in items["methods"].items():
                html += self.generate_table_row(name, info)
        else:
            # This is a function table
            for name, info in items.items():
                html += self.generate_table_row(name, info)

        html += "</table>"
        return html

    def generate_table_row(self, name, info):
        short_name = name.split(".")[-1]
        if short_name[0] == "_":
            return ""
        # first part of name before first . is replaced with "tskit" due to * imports
        name = "tskit." + name.split(".", 1)[1]
        in_sphinx = name in self.doc_items
        highlight = (
            ' class="highlight"' if info["has_docstring"] and not in_sphinx else ""
        )
        highlight = (
            ' class="highlight-low"'
            if not info["has_docstring"] and not in_sphinx
            else highlight
        )
        return f"""
        <tr{highlight}>
            <td>{short_name}</td>
            <td>{info['type']}</td>
            <td>{'<span class="tick">✓</span>' if info['has_docstring'] else
                 '<span class="cross">✗</span>'}</td>
            <td>{'<span class="tick">✓</span>' if in_sphinx else
                 '<span class="cross">✗</span>'}</td>
        </tr>
        """

    def run(self):
        self.analyze_source()
        self.analyze_docs()
        return self.generate_html_report()


analyzer = DocAnalyzer("python/tskit", "docs/_build/html")
report = analyzer.run()

with open("documentation_report.html", "w") as f:
    f.write(report)

print("Report generated as documentation_report.html")
