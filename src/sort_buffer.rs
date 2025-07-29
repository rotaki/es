// In-memory sort buffer implementation

pub struct SortBufferImpl {
    data: Vec<(Vec<u8>, Vec<u8>)>,
    memory_used: usize,
    memory_limit: usize,
}

impl SortBufferImpl {
    pub fn new(memory_limit: usize) -> Self {
        Self {
            data: Vec::new(),
            memory_used: 0,
            memory_limit,
        }
    }
}

impl super::SortBuffer for SortBufferImpl {
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    fn has_space(&self, key: &[u8], value: &[u8]) -> bool {
        let entry_size = key.len() + value.len() + std::mem::size_of::<u32>() * 2; // key and value lengths
        self.memory_used + entry_size <= self.memory_limit
    }

    fn append(&mut self, key: Vec<u8>, value: Vec<u8>) -> bool {
        let entry_size = key.len() + value.len() + std::mem::size_of::<u32>() * 2; // key and value lengths

        if self.memory_used + entry_size > self.memory_limit {
            return false;
        }

        self.data.push((key, value));
        self.memory_used += entry_size;
        true
    }

    fn sort(&mut self) {
        self.data.sort_by(|a, b| a.0.cmp(&b.0));
    }

    fn drain(&mut self) -> Box<dyn Iterator<Item = (Vec<u8>, Vec<u8>)>> {
        let data = std::mem::take(&mut self.data);
        self.memory_used = 0;
        Box::new(data.into_iter())
    }

    fn reset(&mut self) {
        self.data.clear();
        self.memory_used = 0;
    }
}
