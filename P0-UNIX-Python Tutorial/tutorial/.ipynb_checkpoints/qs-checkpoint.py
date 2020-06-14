def quicksort(l):
    if len(l) <= 1:
        return l
    else:
        p = l[0]
        return quicksort([ _ for _ in l if _>p ]) + [p] + quicksort([ _ for _ in l if _<p ])
        

import random
l =list(range(10))
random.shuffle(l)
print(f'before:{l}, after:{quicksort(l)}')