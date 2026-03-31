def other := "theorem"

def other_value := 2+5
-- theorem add_comm (a b : Nat) : a + b = b + a := by
--   exact Nat.add_comm a b
-- 
-- #check add_comm
-- #check add_comm 1 2

-- #check 2 + 2 = 4

def FermatLastTheorem :=
    ∀ x y z n : Nat, n > 2 ∧ x * y * z ≠ 0 → x ^ n + y ^ n ≠ z ^ n

def α : Type := Nat
def β : Type := Bool

#check Prod α β
#check FermatLastTheorem
#eval other_value
#check other_value
