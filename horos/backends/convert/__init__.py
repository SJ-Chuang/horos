"""Model-format converters shared by backends (E8-T3).

Only this package imports the conversion toolchains (onnx2tf / tensorflow /
ai_edge_litert), lazily and inside functions — R1: nothing above
horos/backends/ knows they exist, and R1b: `import horos` never loads them.
"""
