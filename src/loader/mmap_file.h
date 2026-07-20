#pragma once

#include <span>
#include <utility>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>
#include <string>
#include <filesystem>

class MmapFile {
private:
    void* map_; // memory-mapped thing
    std::size_t size_; // file size
    int fd_; // file descriptor
public:
    MmapFile() = default;
    explicit MmapFile(const std::filesystem::path& filepath) : map_(MAP_FAILED), size_(0), fd_(-1) {
        fd_ = open(filepath.c_str(), O_RDONLY);
        if (fd_ == -1) throw std::runtime_error("Failed to open file");

        struct stat st;
        if (fstat(fd_, &st) == -1) {
            close(fd_);
            throw std::runtime_error("Failed to get file size");
        }

        size_ = st.st_size;
        if (size_ == 0) {
            close(fd_);
            throw std::runtime_error("Cannot map empty file");
        }

        map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (map_ == MAP_FAILED) {
            close(fd_);
            throw std::runtime_error("mmap failed");
        }
    }

    ~MmapFile() {
        if (map_ != MAP_FAILED) munmap(map_, size_);
        if (fd_ != -1) close(fd_);
    }

    // Copy
    MmapFile(const MmapFile&) = delete;
    MmapFile& operator=(const MmapFile&) = delete;

    // Move
    MmapFile(MmapFile&& moved) noexcept {
        map_ = std::exchange(moved.map_, MAP_FAILED);
        size_ = std::exchange(moved.size_, 0);
        fd_ = std::exchange(moved.fd_, -1);
    }
    MmapFile& operator=(MmapFile&& o) noexcept;

    std::span<const std::byte> get_bytes() const noexcept {
        return {static_cast<const std::byte*>(map_), size_};
    }
    const char* get_data() const noexcept { return static_cast<const char*>(map_); } // shouldn't need write so not gonna write an overload
    size_t get_size() const noexcept { return size_; }
};

int map_file();