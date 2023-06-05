# Extract the optimized Z-matrix from Q-Chem output logs.
# Zilin Song, 2023 Mar 27
# 

def extract_zmat(i: int):
  """extract reaction i."""
  qclog = open(f"./ms0_car/fig_pyscf/baker_set_reactants/rxn{i}/qchem_baker_set_{i}.qclog", 'r')
  lines = qclog.readlines()

  entered_region = False
  located_line   = False

  xyz_lines = []

  for l in lines:
    if l.startswith(' **  OPTIMIZATION CONVERGED  **'):
      entered_region = True

    if entered_region == True and l.startswith('     ATOM                X               Y               Z'):
      located_line = True
    
    if l.startswith('Z-matrix Print:'):
      located_line = False
      entered_region = False

    if entered_region == True and located_line == True:
      xyz_lines.append(l.strip())

  with open(f'./ms0_car/fig_pyscf/end_points/rxn{i}/reactant.xyz', 'w') as f:

    f.write(f"{len(xyz_lines[1:])-1}\n")
    f.write(f'baker set rxn {i} reactant.\n')

    for l in xyz_lines[1:]:
      words = l.split()[1:]
      
      if len(words)==4:
        f.write(f'{words[0]:<4} {words[1]:>16} {words[2]:>16} {words[3]:>16}\n')
      

if __name__ == "__main__":
  for i in [3]: # [1,3,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]:
    extract_zmat(i)
