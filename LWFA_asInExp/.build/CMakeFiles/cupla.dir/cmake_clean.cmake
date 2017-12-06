file(REMOVE_RECURSE
  "libcupla.pdb"
  "libcupla.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/cupla.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
