import Lake
open Lake DSL

package «rqe_proof» where
  -- add package configuration options here

lean_lib «RQEProof» where
  -- add library configuration options here

@[default_target]
lean_exe «rqe_proof» where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"
