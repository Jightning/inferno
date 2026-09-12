#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "commands/bench.h"

namespace {

bench::Row sample_row() {
    return bench::Row{"a1b2c3d", "2026-08-12", "fp32-nocache", 1, 12.4, 0.9, 2050.0, "M9 baseline"};
}

// A results.csv living in a unique temp path, removed with the test.
class TempCsv {
public:
    TempCsv() {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() / ("inferno_test_bench_" + std::to_string(counter++)) / "results.csv";
    }

    ~TempCsv() {
        std::error_code ec;
        std::filesystem::remove_all(path_.parent_path(), ec);
    }

    TempCsv(const TempCsv&) = delete;
    TempCsv& operator=(const TempCsv&) = delete;

    std::string path() const { return path_.string(); }

    void touch_empty() const {
        std::filesystem::create_directories(path_.parent_path());
        std::ofstream out(path_);
    }

    std::vector<std::string> lines() const {
        std::vector<std::string> out;
        std::ifstream in(path_);
        for (std::string line; std::getline(in, line);) out.push_back(line);
        return out;
    }

private:
    std::filesystem::path path_;
};

}  // namespace

TEST_CASE("median takes the middle value, not the mean") {
    // The point of the median: one slow outlier must not move the result.
    CHECK(bench::median({10.0, 11.0, 100.0}) == doctest::Approx(11.0));
    CHECK(bench::median({7.5}) == doctest::Approx(7.5));
    CHECK(bench::median({4.0, 2.0}) == doctest::Approx(3.0));  // even: midpoint
    CHECK_THROWS(bench::median({}));
}

TEST_CASE("format_csv_row matches the schema") {
    CHECK(bench::format_csv_row(sample_row()) == "a1b2c3d,2026-08-12,fp32-nocache,1,12.40,0.90,2050,\"M9 baseline\"\n");
}

TEST_CASE("notes are quoted so commas and quotes cannot break the columns") {
    bench::Row row = sample_row();
    row.notes = "before AVX2, \"naive\" scalar";

    const std::string line = bench::format_csv_row(row);
    CHECK(line.find(",\"before AVX2, \"\"naive\"\" scalar\"\n") != std::string::npos);
}

TEST_CASE("a config name that would corrupt the columns is rejected") {
    bench::Row row = sample_row();
    row.config = "fp32,nocache";
    CHECK_THROWS(bench::format_csv_row(row));
}

TEST_CASE("append_csv_row writes a header once, then only rows") {
    const TempCsv csv;

    bench::append_csv_row(csv.path(), sample_row());  // parent directory does not exist yet

    bench::Row second = sample_row();
    second.config = "fp32-kv";
    bench::append_csv_row(csv.path(), second);

    const std::vector<std::string> lines = csv.lines();
    REQUIRE(lines.size() == 3);
    CHECK(lines[0] == bench::kCsvHeader);
    CHECK(lines[1].find("fp32-nocache") != std::string::npos);
    CHECK(lines[2].find("fp32-kv") != std::string::npos);
}

TEST_CASE("an existing but empty results file still gets a header") {
    // benchmarks/results.csv is committed empty, so "exists" alone must not skip the header.
    const TempCsv csv;
    csv.touch_empty();

    bench::append_csv_row(csv.path(), sample_row());

    const std::vector<std::string> lines = csv.lines();
    REQUIRE(lines.size() == 2);
    CHECK(lines[0] == bench::kCsvHeader);
}

TEST_CASE("the git hash is marked when the working tree does not match it") {
    const std::string hash = bench::current_git_hash();
    CHECK(!hash.empty());
    // Either a real short hash, "unknown" outside a repo, or explicitly flagged dirty.
    CHECK((hash == "unknown" || hash.find_first_of(" \t\n") == std::string::npos));
}

TEST_CASE("current_date is an ISO day") {
    const std::string date = bench::current_date();
    REQUIRE(date.size() == 10);
    CHECK(date[4] == '-');
    CHECK(date[7] == '-');
}
