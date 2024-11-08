//! Rust bindings of llama.cpp

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use miette::{bail, Severity};
use std::ffi::{c_char, CStr, CString};
use std::fmt;
use std::rc::Rc;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

type Result<T> = miette::Result<T>;

type Token = llama_token;

pub struct LlamaModel {
    pimpl: *mut llama_model,
}

impl LlamaModel {
    /// Load the model from 🤗 Hub.
    pub async fn from_hf<S: AsRef<str>>(repo: S, model: S, path_model: S) -> Result<Self> {
        todo!()
    }

    pub async fn from_url<S: AsRef<str>>(url: S) -> Result<Self> {
        todo!()
    }

    /// Load a model from the file.
    pub fn from_file<S: AsRef<str>>(path_model: S, params: &LlamaModelParams) -> Result<Self> {
        let c_path_model = CString::new(path_model.as_ref()).unwrap();
        let pimpl = unsafe { llama_load_model_from_file(c_path_model.as_ptr(), params.pimpl) };
        if pimpl.is_null() {
            bail!(severity = Severity::Error, "unable to load model")
        } else {
            Ok(LlamaModel { pimpl })
        }
    }

    // Returns the description of the model type
    pub fn description(&self) -> String {
        let mut desc = [0 as c_char; 256];
        unsafe {
            llama_model_desc(self.pimpl, desc.as_mut_ptr(), desc.len());
            CStr::from_ptr(desc.as_ptr()).to_str().unwrap().to_string()
        }
    }

    /// Returns the total size of all tensors in the model in in bytes.
    pub fn model_size(&self) -> usize {
        unsafe { llama_model_size(self.pimpl) as usize }
    }

    /// Returns the total number of parameters in the model.
    pub fn num_params(&self) -> usize {
        unsafe { llama_model_n_params(self.pimpl) as usize }
    }

    /// Returns true if the model contains an encoder.
    pub fn has_encoder(&self) -> bool {
        unsafe { llama_model_has_encoder(self.pimpl) }
    }

    /// Returns true if the model contaisn a decoder.
    pub fn has_decoder(&self) -> bool {
        unsafe { llama_model_has_decoder(self.pimpl) }
    }
}

impl fmt::Display for LlamaModel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(&self.description())
    }
}

impl Drop for LlamaModel {
    fn drop(&mut self) {
        unsafe {
            llama_free_model(self.pimpl);
        }
    }
}

pub struct LlamaContext {
    pimpl: *mut llama_context,
}

impl LlamaContext {
    fn new(model: &LlamaModel, params: &LlamaContextParams) -> Result<Self> {
        unsafe {
            let pimpl = llama_new_context_with_model(model.pimpl, params.pimpl);
            if pimpl.is_null() {
                bail!(severity = Severity::Error, "unable to create llama_context")
            } else {
                Ok(LlamaContext { pimpl })
            }
        }
    }
}

impl Drop for LlamaContext {
    fn drop(&mut self) {
        unsafe {
            llama_free(self.pimpl);
        }
    }
}

pub struct LlamaModelParams {
    pimpl: llama_model_params,
}

pub struct LlamaContextParams {
    pimpl: llama_context_params,
}

impl LlamaContextParams {
    pub fn default() -> Self {
        todo!()
        // LlamaContextParams {
        //     pimpl: unsafe { llama_context_default_params() }
        // }
    }
}

pub struct LlamaTokenizer {
    model: Rc<LlamaModel>,
}

impl LlamaTokenizer {
    pub fn from_model(model: Rc<LlamaModel>) -> Self {
        LlamaTokenizer {
            model: model.clone(),
        }
    }
}

impl LlamaTokenizer {
    /// Conver the text into tokens.
    ///
    /// # Arguments
    ///
    /// * `text`
    /// * `add_special`
    /// * `parse_special`
    ///
    /// # Example
    pub fn encode<S: AsRef<str>>(&self, text: S, add_special: bool, parse_special: bool) -> Result<Vec<i32>> {
        // Find the number of tokens
        let c_text = CString::new(text.as_ref()).unwrap();
        let n_prompt = unsafe { llama_tokenize(self.model.pimpl, c_text.as_ptr(), text.as_ref().len() as i32, std::ptr::null_mut(), 0, add_special, parse_special) };

        let mut tokens = vec![0, n_prompt];
        let rc = unsafe { llama_tokenize(self.model.pimpl, c_text.as_ptr(), text.as_ref().len() as i32, tokens.as_mut_ptr(), n_prompt, add_special, parse_special) };
        if rc < 0 {
            bail!("failed to tokenize prompt")
        } else {
            unsafe { tokens.set_len(n_prompt as usize); }
            Ok(tokens)
        }
    }

    ///
    pub fn decode(&self, tokens: Vec<Token>, remove_special: bool, unparse_special: bool) -> Result<Vec<Token>> {
        todo!()
    }

    ///
    pub fn convert_token_to_ids(&self) -> Result<Token> {
        todo!()
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn test_model_from_file() {
        // let model = LlamaModel::from_file("".into());
    }
}
