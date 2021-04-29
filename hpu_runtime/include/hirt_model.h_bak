#ifndef LIBHIRT_MODEL_H__
#define LIBHIRT_MODEL_H__

#include "hirt.h"
#include "hirt_cqueue.h"

//#ifdef __cplusplus
//extern "C" {
//#endif // __cplusplus

#define HIRT_MODEL_MEMINDEX_PARAM           (0)
#define HIRT_MODEL_MEMINDEX_INPUT           (1)
#define HIRT_MODEL_MEMINDEX_OUTPUT          (2)
#define HIRT_MODEL_MEMINDEX_WEIGHT          (3)
#define HIRT_MODEL_MEMINDEX_BIAS            (4)
#define HIRT_MODEL_MEMINDEX_FEATURE         (5)
#define HIRT_MODEL_MEMINDEX_LUT             (6)
#define HIRT_MODEL_MEMINDEX_MAX             (7)

using namespace hisdk;

class hirtModelCtx
{
public:
    class Task  {
    public:
        Task() { }
        Task(const TaskListEntry &e) : mEntry(e) { }
        Task(const Task &o) : mEntry(o.mEntry) { }
        u16_t id() const { return mEntry.id; }
        u32_t interface() const { return mEntry.interface; }
        //i32_t instance()  const { return mEntry.instance; }
        std::vector<u32_t> &pr_addr_list() { return mEntry.pr_addr_list; }
        std::vector<u32_t> &in_addr_list() { return mEntry.in_addr_list; }
        std::vector<u32_t> &ou_addr_list() { return mEntry.ou_addr_list; }
        std::vector<u32_t> &wt_addr_list() { return mEntry.wt_addr_list; }
        std::vector<u32_t> &bs_addr_list() { return mEntry.bs_addr_list; }
        std::vector<u32_t> &fm_addr_list() { return mEntry.fm_addr_list; }
        std::vector<u32_t> &lu_addr_list() { return mEntry.lu_addr_list; }
    protected:
        TaskListEntry mEntry;

    };

    class Memory {
    public:
        Memory() : hMem(0) { }
        Memory(const MemoryListEntry &e) : hMem(0), mEntry(e) { }
        Memory(const Memory &o) : hMem(o.hMem), mEntry(o.mEntry) { }
        inline u16_t id() { return mEntry.id; }
        inline u64_t size() { return mEntry.size; }
        inline u32_t alignment() { return mEntry.alignment; }
        inline u32_t domain() { return mEntry.domain; }
        inline u32_t flags() { return mEntry.flags; }
        
        inline void setHandle(hirtGMemAddress_t h) { hMem = h; }
        inline hirtGMemAddress_t getHandle() const { return hMem; }
        
        inline std::vector<std::string> & contents() { return mEntry.contents; }
        inline std::vector<u64_t> & offsets() { return mEntry.offsets; }
   #if 0
        inline int inputBindId() const {
            if ( mEntry.flags & mEntry.flags_input() ) {
                return (int) mEntry.bind_id;
            }
            return -1;
        }
        inline int outputBindId() const {
            if ( mEntry.flags & mEntry.flags_output() ) {
                return (int) mEntry.bind_id;
            }
            return -1;
        }
        inline bool bindable() const {
            return !!(mEntry.flags & (mEntry.flags_input() | mEntry.flags_output()));
        }
        inline int bindId(IOD &which) const
        {
            // there should be only one valid.  but if not take in order of
            // input, output
            if ( mEntry.flags & mEntry.flags_input() ) {
                which = IOD_Input;
                return (int) mEntry.bind_id;
            } else if ( mEntry.flags & mEntry.flags_output() ) {
                which = IOD_Output;
                return (int) mEntry.bind_id;
            }
            which = IOD_Max;
            return -1;
        };
        inline int tensorDescId() const {
            if ( mEntry.flags & ( mEntry.flags_input() | mEntry.flags_output() ) ) {
                return (int) mEntry.tensor_desc_id;
            }
            return -1;
        }
    #endif
    protected:
        hirtGMemAddress_t hMem;
        MemoryListEntry mEntry;
    };

    class Address {
    public:
        Address() { }
        Address(const AddressListEntry &e) : mEntry(e) { }
        Address(const Address &o) : mEntry(o.mEntry) { }
        u16_t id() const { return mEntry.id; }
        u16_t mem_id() const { return mEntry.mem_id; }
        u64_t size()   const { return mEntry.size; }
        u64_t offset() const { return mEntry.offset; }
    public:
        AddressListEntry mEntry;
    };

protected:
    std::vector<TaskListEntry>    m_task_entries;
    std::vector<MemoryListEntry>  m_memory_entries;
    std::vector<AddressListEntry> m_address_entries;

    std::vector<Task>    m_task;
    std::vector<Memory>  m_memory;
    std::vector<Address> m_address;

    Memory m_memory_input;
    Memory m_memory_output;

public:
    hirtModelCtx();
    ~hirtModelCtx();

    hisdkRet_t loadModel(const char* fname);
    void       unloadModel();
    hisdkRet_t loadMemories();
    void       unloadMemories() {};
    hisdkRet_t submit(const unsigned char* pBufInput, unsigned char* pBufOutput, hirtCmdQueue_t *pCmdQueue);
    u64_t getInputTensorBufferSize(void);
    u64_t getOutputTensorBufferSize(void);
    void debugTaskEntrys(TaskListEntry &task);

private:
    hModel m_hModel;

};

//#ifdef __cplusplus
//}
//#endif
#endif /*LIBHIRT_MODEL_H__*/

