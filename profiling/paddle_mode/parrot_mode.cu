#include <iostream>
#include "parrot.hpp"

/**
 * @brief Compute mode (most frequent value) and its original index using parrot
 * @param data 2D array reshaped as (num_rows, num_cols)
 * @param num_rows Number of rows to process
 * @param num_cols Number of columns per row
 * @return Array of pairs containing (mode_value, original_index) for each row
 */
template <typename Array>
auto GetModeBySort_Parrot(const Array& data, int num_rows, int num_cols) {
    using T = typename Array::value_type;

    std::vector<thrust::pair<T, int>> results;
    results.reserve(num_rows);

    for (int r = 0; r < num_rows; ++r) {
        auto mode  = parrot::stats::mode(data.row(r)).value();
        auto index = data.row(r).last_index_of(mode);
        results.push_back(thrust::make_pair(mode, index));
    }

    // Convert results to parrot array
    return parrot::array(results);
}

void test_parrot_mode() {
    std::cout << "Testing Parrot Mode Computation\n";
    std::cout << "===============================\n";

    const int num_rows = 500;
    const int num_cols = 300;

    // Generate test data: random values from 1 to 10 for better mode detection
    std::vector<float> test_data;
    test_data.reserve(num_rows * num_cols);

    // Use a simple pattern to ensure some repeated values for mode calculation
    for (int i = 0; i < num_rows * num_cols; ++i) {
        test_data.push_back(static_cast<float>((i % 10) + 1));  // Values 1-10
    }

    // Create parrot array and reshape to 2D
    auto data = parrot::array(test_data).reshape({num_rows, num_cols});

    std::cout << "Input data (reshaped to " << num_rows << "x" << num_cols
              << "):\n";
    std::cout << "Data contains values from 1 to 10 in a cyclic pattern.\n";
    std::cout << "Total elements: " << (num_rows * num_cols) << "\n";

    // Test full mode computation (values + original indices)
    std::cout << "\nFull mode computation (values + original indices):\n";
    std::cout << "Processing " << num_rows << " rows with " << num_cols
              << " columns each...\n";

    auto full_modes = GetModeBySort_Parrot(data, num_rows, num_cols);
    auto results    = full_modes.to_host();

    // Show results for first 10 rows only
    const int display_rows = std::min(10, num_rows);
    std::cout << "Showing results for first " << display_rows << " rows:\n";
    for (int i = 0; i < display_rows; ++i) {
        std::cout << "Row " << i << " - Mode: " << results[i].first
                  << ", Original Index: " << results[i].second << "\n";
    }

    // Basic validation - ensure all modes are in expected range [1, 10]
    bool all_valid = true;
    for (int i = 0; i < num_rows; ++i) {
        if (results[i].first < 1.0f || results[i].first > 10.0f) {
            std::cout << "ERROR: Row " << i
                      << " has invalid mode: " << results[i].first << "\n";
            all_valid = false;
            break;
        }
    }

    if (all_valid) {
        std::cout << "\n✓ All mode computations completed successfully!\n";
        std::cout << "✓ All modes are within expected range [1, 10]\n";
    } else {
        std::cout << "\n✗ Some mode computations failed validation!\n";
    }
}

int main() {
    std::cout << "Parrot Mode Computation Example\n";
    std::cout << "==============================\n\n";

    try {
        test_parrot_mode();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}