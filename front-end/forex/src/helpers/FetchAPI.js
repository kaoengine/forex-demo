export const fetchAPI = (url, params) => {
    if(params) {
        let newURL = url + Object.keys(params).map(k =>k+ '=' + params[k]);
        return fetch(newURL);
    }
    return fetch(url)
}