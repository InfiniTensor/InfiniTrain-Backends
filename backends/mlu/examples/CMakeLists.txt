function(add_mlu_example target_name output_name)
  add_executable(${target_name} ${ARGN})
  # Reuse the upstream example sources while injecting this provider's
  # declaration and registration callback.
  target_compile_definitions(${target_name} PRIVATE
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER=\"infini_train_mlu/backend.h\"
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_REGISTRAR=infini_train::backends::mlu::RegisterBackend
  )
  target_link_libraries(${target_name} PRIVATE
    InfiniTrain::Backend::MLUExecutable)
  set_target_properties(${target_name} PROPERTIES
    OUTPUT_NAME ${output_name}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endfunction()

# GPT-2 is the only model target for the current MLU stage. The sources come
# straight from the pinned InfiniTrain tree; nothing is copied or patched here.
add_mlu_example(mlu_gpt2 gpt2
  "${INFINITRAIN_SOURCE_DIR}/example/gpt2/main.cc"
  "${INFINITRAIN_SOURCE_DIR}/example/gpt2/checkpoint_loader.cc"
  "${INFINITRAIN_SOURCE_DIR}/example/common/tiny_shakespeare_dataset.cc"
  "${INFINITRAIN_SOURCE_DIR}/example/common/tokenizer.cc"
  "${INFINITRAIN_SOURCE_DIR}/example/common/utils.cc"
)
