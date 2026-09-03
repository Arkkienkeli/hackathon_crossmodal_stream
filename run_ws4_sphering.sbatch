#!/bin/bash
#SBATCH --job-name=ws4_sphere
#SBATCH --account=project_2010376
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=ws4_sphere_%j.out

cd /scratch/project_2010376/cellpainting2026_ws4/hackathon_crossmodal_stream

./.venv/bin/python ws4_spherize_openscreen.py
