# TopoFluid2D

A 2D fluid simulation using Taichi with Voronoi diagram-based topology optimization.

## Description

This project implements a 2D fluid simulation that combines:
- Fluid dynamics simulation using Taichi
- Voronoi diagram-based topology
- Shape optimization capabilities
- Particle system for visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/arthifact/TopoFluid2D.git
cd TopoFluid2D
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

There are two main simulation files:

1. Basic simulation:
```bash
python TaichiVoro_2.py
```

2. Advanced simulation with particle system:
```bash
python taichivoro_yes.py
```

The simulation uses shape data from the `shapes/` directory. By default, it uses the 2D bunny shape from `shapes/bunny2D.csv`.

## Shape Configuration

You can modify the shape parameters in the Python files:
- `SHAPE_MODE`: Type of shape input (currently supports "CSV")
- `SHAPE_SCALE`: Adjust the size of the shape
- `SHAPE_CENTER`: Move the shape's position

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
