/*
 * Author: Alejandro Valencia
 * Update: 24 March, 2023
 */

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <optional>

int main()
{
  // Given

#ifdef USE_DEFAULT_CONFIG
  const std::string file_name{"using_select/primary_input.json"};
#else
  const std::string file_name{"using_select/secondary_input.json"};
#endif

  std::optional<std::ifstream> input_file_(file_name);

  if (!input_file_.has_value())
  {
    std::cout << "Input file has no value"
              << "\n";
    return 1;
  }

  const auto input_data = nlohmann::json::parse(input_file_.value());
  std::cout << input_data["name"] << "\n";
  return 0;
}
