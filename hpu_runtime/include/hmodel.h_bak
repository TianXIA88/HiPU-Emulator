#ifndef _HISDK_HMODEL_H__
#define _HISDK_HMODEL_H__

#include <map>
#include <vector>
#include "hmodel_generated.h"


#define HISDK_HMODEL_INTERFACE_NONE 0U
#define HISDK_HMODEL_INTERFACE_GPU 1U
#define HISDK_HMODEL_INTERFACE_CPU 2U

#define HISDK_HMODEL_SUB_INTERFACE_DLA1_NONE  0U
#define HISDK_HMODEL_SUB_INTERFACE_DLA1_ADDR0 1U
#define HISDK_HMODEL_SUB_INTERFACE_DLA1_DEPS  2U
#define HISDK_HMODEL_SUB_INTERFACE_DLA1_OPS   3U
#define HISDK_HMODEL_SUB_INTERFACE_DLA1_SURFS 4U
#define HISDK_HMODEL_SUB_INTERFACE_DLA1_LUTS  5U

#define HISDK_HMODEL_SUB_INTERFACE_EMU1_NONE  0U
#define HISDK_HMODEL_SUB_INTERFACE_EMU1_ADDR0 1U
/* #define HISDK_HMODEL_SUB_INTERFACE_EMU1_DEPS  2U */
#define HISDK_HMODEL_SUB_INTERFACE_EMU1_OPS   3U
#define HISDK_HMODEL_SUB_INTERFACE_EMU1_SURFS 4U
/* #define HISDK_HMODEL_SUB_INTERFACE_EMU1_LUTS  5U */


#define HISDK_HMODEL_MEMORY_DOMAIN_GMEM 0U
#define HISDK_HMODEL_MEMORY_DOMAIN_MRAB 1U

#define HISDK_HMODEL_MEMORY_FLAGS_NONE    0U
#define HISDK_HMODEL_MEMORY_FLAGS_ALLOC   1U
#define HISDK_HMODEL_MEMORY_FLAGS_SET     2U
#define HISDK_HMODEL_MEMORY_FLAGS_INPUT   4U
#define HISDK_HMODEL_MEMORY_FLAGS_OUTPUT  8U
#define HISDK_HMODEL_MEMORY_FLAGS_DEBUG  16U

#define HISDK_HMODEL_EVENT_OP_WAIT   0U
#define HISDK_HMODEL_EVENT_OP_SIGNAL 1U

#define HISDK_HMODEL_TENSOR_DESC_NUM_STRIDES 8U    /* a little room to grow */


namespace hisdk
{

enum Interface {
    Interface_NONE = HISDK_HMODEL_INTERFACE_NONE,
    Interface_GPU  = HISDK_HMODEL_INTERFACE_GPU,
    Interface_CPU  = HISDK_HMODEL_INTERFACE_CPU,
};

enum MemoryDomain {
    MemoryDomain_GMEM   = HISDK_HMODEL_MEMORY_DOMAIN_GMEM,
    MemoryDomain_MRAB   = HISDK_HMODEL_MEMORY_DOMAIN_MRAB,
};

enum MemoryFlags {
    MemoryFlags_NONE   = HISDK_HMODEL_MEMORY_FLAGS_NONE,
    MemoryFlags_ALLOC  = HISDK_HMODEL_MEMORY_FLAGS_ALLOC,
    MemoryFlags_SET    = HISDK_HMODEL_MEMORY_FLAGS_SET,
    MemoryFlags_INPUT  = HISDK_HMODEL_MEMORY_FLAGS_INPUT,
    MemoryFlags_OUTPUT = HISDK_HMODEL_MEMORY_FLAGS_OUTPUT,
    MemoryFlags_DEBUG  = HISDK_HMODEL_MEMORY_FLAGS_DEBUG
};

struct Version
{
    uint8_t main;
    uint8_t sub;
    uint8_t sub_minor;
    Version(uint8_t maj, uint8_t min, uint8_t sub) : main(maj), sub(min), sub_minor(sub) { }
    Version() : main(0), sub(0), sub_minor(0) { }
};

struct MemoryListEntry
{
    uint32_t id;
    uint64_t size;
    uint32_t alignment; // 0 for n/a, otherwise byte alignment
    uint32_t  domain;
    static inline uint32_t domain_gmem() { return MemoryDomain_GMEM; }
    static inline uint32_t domain_mrab() { return MemoryDomain_MRAB; }
    uint32_t  flags; // alloc or alloc_content or is-input or is-output
    static inline uint32_t  flags_alloc()  { return MemoryFlags_ALLOC;  }
    static inline uint32_t  flags_set()    { return MemoryFlags_SET;    }
    static inline uint32_t  flags_input()  { return MemoryFlags_INPUT;  }
    static inline uint32_t  flags_output() { return MemoryFlags_OUTPUT; }
    static inline uint32_t  flags_debug()  { return MemoryFlags_DEBUG;  }
    std::vector<std::string> contents;  // symbolic reference to content blob
    std::vector<uint64_t>    offsets;   // associated offset for contents

    MemoryListEntry() : id(0), size(0), alignment(0), domain(0), flags(0),
                        contents(), offsets() { }
    MemoryListEntry(const MemoryListEntry &o) : id(o.id), size(o.size), alignment(o.alignment), domain(o.domain), flags(o.flags),
                                                contents(o.contents),
                                                offsets(o.offsets) { }
    MemoryListEntry(uint32_t i, uint64_t s, uint32_t a, uint32_t d, uint32_t f, std::string sym = std::string(), uint64_t o = 0) :
        id(i), size(s), alignment(a), domain(d), flags(f)
    {
        if ( sym.size() )
        {
            contents.push_back(sym);
            offsets.push_back(o);
        }
    }
    void clear()
    {
        contents.clear();
        offsets.clear();
    }
};

struct TaskListEntry
{
    uint32_t id;
    std::string name;
    std::string type;
    
    uint32_t interface; // DLA interface id
    static inline uint32_t interface_NONE() { return Interface_NONE; }
    static inline uint32_t interface_GPU()  { return Interface_GPU;  }
    static inline uint32_t interface_CPU()  { return Interface_CPU;  }

    std::vector<uint32_t> pr_addr_list;
    std::vector<uint32_t> in_addr_list;
    std::vector<uint32_t> ou_addr_list;
    std::vector<uint32_t> wt_addr_list;
    std::vector<uint32_t> bs_addr_list;
    std::vector<uint32_t> fm_addr_list;
    std::vector<uint32_t> lu_addr_list;
    TaskListEntry(const TaskListEntry &o) :
        name(o.name),
        type(o.type),
        id(o.id),
        interface(o.interface),
        pr_addr_list(o.pr_addr_list),
        in_addr_list(o.in_addr_list),
        ou_addr_list(o.ou_addr_list),
        wt_addr_list(o.wt_addr_list),
        bs_addr_list(o.bs_addr_list),
        fm_addr_list(o.fm_addr_list),
        lu_addr_list(o.lu_addr_list) { }

    TaskListEntry() : id(0),
                      interface(Interface_NONE),
                      pr_addr_list(),
                      in_addr_list(),
                      ou_addr_list(),
                      wt_addr_list(),
                      bs_addr_list(),
                      fm_addr_list(),
                      lu_addr_list() { }
    TaskListEntry(uint32_t i) : id(i),
                      interface(Interface_NONE),
                      pr_addr_list(),
                      in_addr_list(),
                      ou_addr_list(),
                      wt_addr_list(),
                      bs_addr_list(),
                      fm_addr_list(),
                      lu_addr_list() { }
    void clear()
    {
        pr_addr_list.clear();
        in_addr_list.clear();
        ou_addr_list.clear();
        wt_addr_list.clear();
        bs_addr_list.clear();
        fm_addr_list.clear();
        lu_addr_list.clear();
    }

};

struct AddressListEntry
{
    uint32_t id;     // all possible address list entries are given an id
    uint32_t mem_id; // determines hRm (+offset from below)
    uint64_t size;   // assert size <= memory[mem_id].size
    uint64_t offset; // assert (offset + size) <= memory[mem_id].size
    AddressListEntry() : id(0), mem_id(0), size(0), offset(0) { }
    AddressListEntry(uint32_t i, uint32_t m, uint64_t s, uint64_t o = 0) : id(i), mem_id(m), size(s), offset(o) { }
    AddressListEntry(const AddressListEntry &o) : id(o.id), mem_id(o.mem_id), size(o.size), offset(o.offset) { }
};

struct Blob
{
    std::string name;
    uint64_t size;
    Interface interface;
    uint32_t type;
    Version version;

    Blob() :
        size(0),
        interface(Interface_NONE),
        type(0) { }

    Blob(const std::string &n, uint64_t s, Interface i, uint32_t si, Version v) :
        name(n),
        size(s),
        interface(i),
        type(si),
        version(v) { }
};

struct Symbol 
{
    std::string name;
    Interface interface;
    uint32_t type;
    Version version;
    uint64_t size;
    uint8_t *data;
    Symbol() { }
};

class hModel
{
public: // externally facing

    virtual std::string getName() const;

    virtual int             getNumMemoryListEntries() const;
    virtual MemoryListEntry getMemoryListEntry(uint32_t pool_id) const;

    virtual int           getNumTaskListEntries() const;
    virtual TaskListEntry getTaskListEntry(uint32_t task_id) const;

    virtual int              getNumAddressListEntries() const;
    virtual AddressListEntry getAddressListEntry(uint32_t address_list_index) const;

public: // internally facing
    int setSymbolContent(std::string name, const Blob &b, uint8_t *data);
    bool getSymbolContent(std::string name, Blob &b, uint8_t *&data);

    void setMemoryListEntries (const std::vector<MemoryListEntry>  &);
    void setTaskListEntries (const std::vector<TaskListEntry>    &);
    void setAddressListEntries (const std::vector<AddressListEntry> &);

    const std::vector<TaskListEntry> & getTaskListEntries()   const;
    const std::vector<MemoryListEntry> & getMemoryListEntries() const;
    const std::vector<AddressListEntry> & getAddressListEntries() const;

    hModel();
    virtual ~hModel();

    virtual bool serialize();
    virtual bool getSerializedData(uint8_t *buffer) const;
    virtual bool getSerializedDataSize(uint64_t *size) const;
    virtual bool deserializeFrom(uint8_t *flatbuf);
protected:
    std::map<std::string, Symbol> mSymbols;
    std::vector<MemoryListEntry>  mMemoryListEntries;
    std::vector<TaskListEntry>    mTaskListEntries;
    std::vector<AddressListEntry> mAddressListEntries;
    std::string mName;

private:
    flatbuffers::FlatBufferBuilder mFbb;
};

} // nvdla

#endif
