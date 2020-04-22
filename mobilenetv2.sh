#SBATCH -p public
#SBATCH -J mobilenetv2_hoplite
#SBATCH -o mn2.out
#SBATCH -e mn2.e%j
#SBATCH --qos general
#SBATCH -n 84
#SBATCH -N 3
#SBATCH -C c6320
#SBATCH -t 12:00:00

rm -rf hoplite
git clone https://github.com/spartan-analysis/hoplite
cd hoplite

module load python/3.6.5
module load cuda/90/toolkit/9.0.176
module load cudnn/7.0/cuda90

python3 -m venv ./env
source ./env/bin/activate
pip3 -r requirements.txt

python mobilenetv2_runner.py
