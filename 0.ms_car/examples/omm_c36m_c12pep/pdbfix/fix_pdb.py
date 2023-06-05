# fix raw pdbs for charmm input
import sys

rawpdb = open(sys.argv[1], 'r')
file_out = open('fixed_{0}'.format(sys.argv[1]), 'w')
segid = sys.argv[2].split('segid=').pop()

file_out.write('REMARK FIX RAW PDBS FOR CHARMM\n')
file_out.write('REMARK ZILIN SONG, AUG 22 2019\n')

lines = rawpdb.readlines()
atomno = 0
for line in lines:

    if line[0:6] == 'HEADER':
        file_out.write('REMARK{0}'.format(line[6:]))

    elif line[0:6] == 'ATOM  ' or line[0:6] == 'HETATM':
        
        if line[16] != ' ' and line[16] != 'A':  continue          # alternative configuration
        if line[21] != segid:                    continue          # chain ID
        if line[17:20] in ['SO4', 'IOD', ' NA']: continue          # skiped compound
        
        atomno += 1

        _b1 = 'ATOM  '                      # 0-5:      LABEL: 'ATOM  '             b1
        _b2 = '{:>5}'.format(atomno)        # 6-10:     ATOM_NUMBER                 b2
        _b3 = ' '                           # 11:       space                       b3
        _b4 = line[12:16]                   # 12-15:    ATOM_NAME                   b4
        _b5 = ' '                           # 16:       ALT_CONF_INDICATOR          b5
        _b6 = line[17:20]                   # 17-19:    RESNAME                     b6
        _b7 = ' '                           # 20:       space                       b7
        _b8 = line[21]                      # 21:       CHAIN_INDENTIFIER           b8
        _b9 = line[22:26]                   # 22-25:    RESID                       b9
        _b10 = ' '                          # 26:       1 space                     b10
        _b11 = '   '                        # 27-29:    3 spaces                    b11
        _b12 = line[30:38]                  # 30-37:    COORDINATE_X                b12
        _b13 = line[38:46]                  # 38-45:    COORDINATE_Y                b13
        _b14 = line[46:54]                  # 46-53:    COORDINATE_Z                b14
        _b15 = line[54:60]                  # 54-59:    OCCUP                       b15
        _b16 = line[60:66]                  # 60-65:    TEMP_FACTOR                 b16
        _b17 = line[66:72]                  # 66-71:    space                       b17
        #_b18 = line[72:76]                  # 72-76:    SEGMENT_ID                  b18   rawpdb => most cases empty
        _b18 = '{:>4}'.format(segid)

        if line[17:20] == 'HOH':
            _b4 = ' OH2'
            _b6 = 'TIP'
            _b7 = '3'
        if line[17:20] == 'HIS':
            _b6 = 'HSD'
        if line[17:20] == 'ILE' and line[12:16] == ' CD1':
            _b4 = ' CD '
        if line[12:16] == ' OXT' or line[12:16] == 'OCT1':
            _b4 = ' OT1'
        if line[12:16] == 'OCT2':
            _b4 = ' OT2'
        if line[17:20] == 'AIX':
            _b6 = 'AMP'

        file_out.write('{0}{1}{2}{3}{4}{5}{6}{7}{8}{9}{10}{11}{12}{13}{14}{15}{16}{17}\n'.format(
            _b1, _b2, _b3, _b4, _b5, _b6, _b7, _b8, _b9, _b10, _b11, _b12, _b13, _b14, _b15, _b16, _b17, _b18))

file_out.write('END\n')
