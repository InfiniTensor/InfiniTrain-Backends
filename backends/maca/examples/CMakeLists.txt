function(add_maca_example target_name output_name)
  add_executable(${target_name} ${ARGN})
  # Reuse upstream example sources while injecting the selected external
  # provider's declaration and registration callback.
  target_compile_definitions(${target_name} PRIVATE
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_HEADER=\"infini_train_maca/backend.h\"
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_REGISTRAR=infini_train::maca::RegisterBackend
    INFINITRAIN_EXAMPLE_EXTERNAL_BACKEND_WORKAROUNDS=FLAGS_maca_multithread_workarounds
  )
  target_link_libraries(${target_name} PRIVATE
    InfiniTrain::Backend::MACAExecutable)
  set_target_properties(${target_name} PROPERTIES
    OUTPUT_NAME ${output_name}
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  )
endfunction()

function(add_maca_model_example model_name)
  set(model_dir "${INFINITRAIN_SOURCE_DIR}/example/${model_name}")
  set(model_sources
    "${INFINITRAIN_SOURCE_DIR}/example/common/tiny_shakespeare_dataset.cc"
    "${INFINITRAIN_SOURCE_DIR}/example/common/utils.cc"
    "${model_dir}/checkpoint_loader.cc"
  )
  if(NOT "${model_name}" STREQUAL "mixtral")
    list(APPEND model_sources
      "${INFINITRAIN_SOURCE_DIR}/example/common/tokenizer.cc")
  endif()

  list(APPEND model_sources "${model_dir}/main.cc")

  add_maca_example("maca_${model_name}" "${model_name}" ${model_sources})
endfunction()

add_maca_model_example(gpt2)
add_maca_model_example(llama3)
add_maca_model_example(mixtral)
