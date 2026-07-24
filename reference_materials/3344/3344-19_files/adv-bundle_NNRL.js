var zo=Object.defineProperty;var $o=(n,e,t)=>e in n?zo(n,e,{enumerable:!0,configurable:!0,writable:!0,value:t}):n[e]=t;var Ae=(n,e,t)=>$o(n,typeof e!="symbol"?e+"":e,t);var be,H,Pi,Yn,Qt,Ei,Li,Oi,kt,ut,gt,Di,ce={},Fi=[],Vo=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i,he=Array.isArray;function Tn(n,e){for(var t in e)n[t]=e[t];return n}function At(n){n&&n.parentNode&&n.parentNode.removeChild(n)}function Mn(n,e,t){var i,o,a,r={};for(a in e)a=="key"?i=e[a]:a=="ref"?o=e[a]:r[a]=e[a];if(arguments.length>2&&(r.children=arguments.length>3?be.call(arguments,2):t),typeof n=="function"&&n.defaultProps!=null)for(a in n.defaultProps)r[a]===void 0&&(r[a]=n.defaultProps[a]);return le(n,r,i,o,null)}function le(n,e,t,i,o){var a={type:n,props:e,key:t,ref:i,__k:null,__:null,__b:0,__e:null,__c:null,constructor:void 0,__v:o??++Pi,__i:-1,__u:0};return o==null&&H.vnode!=null&&H.vnode(a),a}function pe(){return{current:null}}function E(n){return n.children}function Cn(n,e){this.props=n,this.context=e}function ne(n,e){if(e==null)return n.__?ne(n.__,n.__i+1):null;for(var t;e<n.__k.length;e++)if((t=n.__k[e])!=null&&t.__e!=null)return t.__e;return typeof n.type=="function"?ne(n):null}function Ni(n){var e,t;if((n=n.__)!=null&&n.__c!=null){for(n.__e=n.__c.base=null,e=0;e<n.__k.length;e++)if((t=n.__k[e])!=null&&t.__e!=null){n.__e=n.__c.base=t.__e;break}return Ni(n)}}function ft(n){(!n.__d&&(n.__d=!0)&&Yn.push(n)&&!Ne.__r++||Qt!==H.debounceRendering)&&((Qt=H.debounceRendering)||Ei)(Ne)}function Ne(){for(var n,e,t,i,o,a,r,l=1;Yn.length;)Yn.length>l&&Yn.sort(Li),n=Yn.shift(),l=Yn.length,n.__d&&(t=void 0,o=(i=(e=n).__v).__e,a=[],r=[],e.__P&&((t=Tn({},i)).__v=i.__v+1,H.vnode&&H.vnode(t),Tt(e.__P,t,i,e.__n,e.__P.namespaceURI,32&i.__u?[o]:null,a,o??ne(i),!!(32&i.__u),r),t.__v=i.__v,t.__.__k[t.__i]=t,Mi(a,t,r),t.__e!=o&&Ni(t)));Ne.__r=0}function Ii(n,e,t,i,o,a,r,l,c,d,p){var u,f,g,x,b,h,m=i&&i.__k||Fi,_=e.length;for(c=Uo(t,e,m,c,_),u=0;u<_;u++)(g=t.__k[u])!=null&&(f=g.__i===-1?ce:m[g.__i]||ce,g.__i=u,h=Tt(n,g,f,o,a,r,l,c,d,p),x=g.__e,g.ref&&f.ref!=g.ref&&(f.ref&&Pt(f.ref,null,g),p.push(g.ref,g.__c||x,g)),b==null&&x!=null&&(b=x),4&g.__u||f.__k===g.__k?c=Ri(g,c,n):typeof g.type=="function"&&h!==void 0?c=h:x&&(c=x.nextSibling),g.__u&=-7);return t.__e=b,c}function Uo(n,e,t,i,o){var a,r,l,c,d,p=t.length,u=p,f=0;for(n.__k=new Array(o),a=0;a<o;a++)(r=e[a])!=null&&typeof r!="boolean"&&typeof r!="function"?(c=a+f,(r=n.__k[a]=typeof r=="string"||typeof r=="number"||typeof r=="bigint"||r.constructor==String?le(null,r,null,null,null):he(r)?le(E,{children:r},null,null,null):r.constructor===void 0&&r.__b>0?le(r.type,r.props,r.key,r.ref?r.ref:null,r.__v):r).__=n,r.__b=n.__b+1,l=null,(d=r.__i=Go(r,t,c,u))!==-1&&(u--,(l=t[d])&&(l.__u|=2)),l==null||l.__v===null?(d==-1&&(o>p?f--:o<p&&f++),typeof r.type!="function"&&(r.__u|=4)):d!=c&&(d==c-1?f--:d==c+1?f++:(d>c?f--:f++,r.__u|=4))):n.__k[a]=null;if(u)for(a=0;a<p;a++)(l=t[a])!=null&&(2&l.__u)==0&&(l.__e==i&&(i=ne(l)),Hi(l,l));return i}function Ri(n,e,t){var i,o;if(typeof n.type=="function"){for(i=n.__k,o=0;i&&o<i.length;o++)i[o]&&(i[o].__=n,e=Ri(i[o],e,t));return e}n.__e!=e&&(e&&n.type&&!t.contains(e)&&(e=ne(n)),t.insertBefore(n.__e,e||null),e=n.__e);do e=e&&e.nextSibling;while(e!=null&&e.nodeType==8);return e}function In(n,e){return e=e||[],n==null||typeof n=="boolean"||(he(n)?n.some(function(t){In(t,e)}):e.push(n)),e}function Go(n,e,t,i){var o,a,r=n.key,l=n.type,c=e[t];if(c===null&&n.key==null||c&&r==c.key&&l===c.type&&(2&c.__u)==0)return t;if(i>(c!=null&&(2&c.__u)==0?1:0))for(o=t-1,a=t+1;o>=0||a<e.length;){if(o>=0){if((c=e[o])&&(2&c.__u)==0&&r==c.key&&l===c.type)return o;o--}if(a<e.length){if((c=e[a])&&(2&c.__u)==0&&r==c.key&&l===c.type)return a;a++}}return-1}function Zt(n,e,t){e[0]=="-"?n.setProperty(e,t??""):n[e]=t==null?"":typeof t!="number"||Vo.test(e)?t:t+"px"}function Te(n,e,t,i,o){var a;n:if(e=="style")if(typeof t=="string")n.style.cssText=t;else{if(typeof i=="string"&&(n.style.cssText=i=""),i)for(e in i)t&&e in t||Zt(n.style,e,"");if(t)for(e in t)i&&t[e]===i[e]||Zt(n.style,e,t[e])}else if(e[0]=="o"&&e[1]=="n")a=e!=(e=e.replace(Oi,"$1")),e=e.toLowerCase()in n||e=="onFocusOut"||e=="onFocusIn"?e.toLowerCase().slice(2):e.slice(2),n.l||(n.l={}),n.l[e+a]=t,t?i?t.t=i.t:(t.t=kt,n.addEventListener(e,a?gt:ut,a)):n.removeEventListener(e,a?gt:ut,a);else{if(o=="http://www.w3.org/2000/svg")e=e.replace(/xlink(H|:h)/,"h").replace(/sName$/,"s");else if(e!="width"&&e!="height"&&e!="href"&&e!="list"&&e!="form"&&e!="tabIndex"&&e!="download"&&e!="rowSpan"&&e!="colSpan"&&e!="role"&&e!="popover"&&e in n)try{n[e]=t??"";break n}catch{}typeof t=="function"||(t==null||t===!1&&e[4]!="-"?n.removeAttribute(e):n.setAttribute(e,e=="popover"&&t==1?"":t))}}function Xt(n){return function(e){if(this.l){var t=this.l[e.type+n];if(e.u==null)e.u=kt++;else if(e.u<t.t)return;return t(H.event?H.event(e):e)}}}function Tt(n,e,t,i,o,a,r,l,c,d){var p,u,f,g,x,b,h,m,_,k,A,S,M,z,O,F,N,B=e.type;if(e.constructor!==void 0)return null;128&t.__u&&(c=!!(32&t.__u),a=[l=e.__e=t.__e]),(p=H.__b)&&p(e);n:if(typeof B=="function")try{if(m=e.props,_="prototype"in B&&B.prototype.render,k=(p=B.contextType)&&i[p.__c],A=p?k?k.props.value:p.__:i,t.__c?h=(u=e.__c=t.__c).__=u.__E:(_?e.__c=u=new B(m,A):(e.__c=u=new Cn(m,A),u.constructor=B,u.render=Wo),k&&k.sub(u),u.props=m,u.state||(u.state={}),u.context=A,u.__n=i,f=u.__d=!0,u.__h=[],u._sb=[]),_&&u.__s==null&&(u.__s=u.state),_&&B.getDerivedStateFromProps!=null&&(u.__s==u.state&&(u.__s=Tn({},u.__s)),Tn(u.__s,B.getDerivedStateFromProps(m,u.__s))),g=u.props,x=u.state,u.__v=e,f)_&&B.getDerivedStateFromProps==null&&u.componentWillMount!=null&&u.componentWillMount(),_&&u.componentDidMount!=null&&u.__h.push(u.componentDidMount);else{if(_&&B.getDerivedStateFromProps==null&&m!==g&&u.componentWillReceiveProps!=null&&u.componentWillReceiveProps(m,A),!u.__e&&(u.shouldComponentUpdate!=null&&u.shouldComponentUpdate(m,u.__s,A)===!1||e.__v==t.__v)){for(e.__v!=t.__v&&(u.props=m,u.state=u.__s,u.__d=!1),e.__e=t.__e,e.__k=t.__k,e.__k.some(function(C){C&&(C.__=e)}),S=0;S<u._sb.length;S++)u.__h.push(u._sb[S]);u._sb=[],u.__h.length&&r.push(u);break n}u.componentWillUpdate!=null&&u.componentWillUpdate(m,u.__s,A),_&&u.componentDidUpdate!=null&&u.__h.push(function(){u.componentDidUpdate(g,x,b)})}if(u.context=A,u.props=m,u.__P=n,u.__e=!1,M=H.__r,z=0,_){for(u.state=u.__s,u.__d=!1,M&&M(e),p=u.render(u.props,u.state,u.context),O=0;O<u._sb.length;O++)u.__h.push(u._sb[O]);u._sb=[]}else do u.__d=!1,M&&M(e),p=u.render(u.props,u.state,u.context),u.state=u.__s;while(u.__d&&++z<25);u.state=u.__s,u.getChildContext!=null&&(i=Tn(Tn({},i),u.getChildContext())),_&&!f&&u.getSnapshotBeforeUpdate!=null&&(b=u.getSnapshotBeforeUpdate(g,x)),F=p,p!=null&&p.type===E&&p.key==null&&(F=Bi(p.props.children)),l=Ii(n,he(F)?F:[F],e,t,i,o,a,r,l,c,d),u.base=e.__e,e.__u&=-161,u.__h.length&&r.push(u),h&&(u.__E=u.__=null)}catch(C){if(e.__v=null,c||a!=null)if(C.then){for(e.__u|=c?160:128;l&&l.nodeType==8&&l.nextSibling;)l=l.nextSibling;a[a.indexOf(l)]=null,e.__e=l}else for(N=a.length;N--;)At(a[N]);else e.__e=t.__e,e.__k=t.__k;H.__e(C,e,t)}else a==null&&e.__v==t.__v?(e.__k=t.__k,e.__e=t.__e):l=e.__e=jo(t.__e,e,t,i,o,a,r,c,d);return(p=H.diffed)&&p(e),128&e.__u?void 0:l}function Mi(n,e,t){for(var i=0;i<t.length;i++)Pt(t[i],t[++i],t[++i]);H.__c&&H.__c(e,n),n.some(function(o){try{n=o.__h,o.__h=[],n.some(function(a){a.call(o)})}catch(a){H.__e(a,o.__v)}})}function Bi(n){return typeof n!="object"||n==null?n:he(n)?n.map(Bi):Tn({},n)}function jo(n,e,t,i,o,a,r,l,c){var d,p,u,f,g,x,b,h=t.props,m=e.props,_=e.type;if(_=="svg"?o="http://www.w3.org/2000/svg":_=="math"?o="http://www.w3.org/1998/Math/MathML":o||(o="http://www.w3.org/1999/xhtml"),a!=null){for(d=0;d<a.length;d++)if((g=a[d])&&"setAttribute"in g==!!_&&(_?g.localName==_:g.nodeType==3)){n=g,a[d]=null;break}}if(n==null){if(_==null)return document.createTextNode(m);n=document.createElementNS(o,_,m.is&&m),l&&(H.__m&&H.__m(e,a),l=!1),a=null}if(_===null)h===m||l&&n.data===m||(n.data=m);else{if(a=a&&be.call(n.childNodes),h=t.props||ce,!l&&a!=null)for(h={},d=0;d<n.attributes.length;d++)h[(g=n.attributes[d]).name]=g.value;for(d in h)if(g=h[d],d!="children"){if(d=="dangerouslySetInnerHTML")u=g;else if(!(d in m)){if(d=="value"&&"defaultValue"in m||d=="checked"&&"defaultChecked"in m)continue;Te(n,d,null,g,o)}}for(d in m)g=m[d],d=="children"?f=g:d=="dangerouslySetInnerHTML"?p=g:d=="value"?x=g:d=="checked"?b=g:l&&typeof g!="function"||h[d]===g||Te(n,d,g,h[d],o);if(p)l||u&&(p.__html===u.__html||p.__html===n.innerHTML)||(n.innerHTML=p.__html),e.__k=[];else if(u&&(n.innerHTML=""),Ii(e.type==="template"?n.content:n,he(f)?f:[f],e,t,i,_=="foreignObject"?"http://www.w3.org/1999/xhtml":o,a,r,a?a[0]:t.__k&&ne(t,0),l,c),a!=null)for(d=a.length;d--;)At(a[d]);l||(d="value",_=="progress"&&x==null?n.removeAttribute("value"):x!==void 0&&(x!==n[d]||_=="progress"&&!x||_=="option"&&x!==h[d])&&Te(n,d,x,h[d],o),d="checked",b!==void 0&&b!==n[d]&&Te(n,d,b,h[d],o))}return n}function Pt(n,e,t){try{if(typeof n=="function"){var i=typeof n.__u=="function";i&&n.__u(),i&&e==null||(n.__u=n(e))}else n.current=e}catch(o){H.__e(o,t)}}function Hi(n,e,t){var i,o;if(H.unmount&&H.unmount(n),(i=n.ref)&&(i.current&&i.current!==n.__e||Pt(i,null,e)),(i=n.__c)!=null){if(i.componentWillUnmount)try{i.componentWillUnmount()}catch(a){H.__e(a,e)}i.base=i.__P=null}if(i=n.__k)for(o=0;o<i.length;o++)i[o]&&Hi(i[o],e,t||typeof n.type!="function");t||At(n.__e),n.__c=n.__=n.__e=void 0}function Wo(n,e,t){return this.constructor(n,t)}function ee(n,e,t){var i,o,a,r;e==document&&(e=document.documentElement),H.__&&H.__(n,e),o=(i=typeof t=="function")?null:t&&t.__k||e.__k,a=[],r=[],Tt(e,n=(!i&&t||e).__k=Mn(E,null,[n]),o||ce,ce,e.namespaceURI,!i&&t?[t]:o?null:e.firstChild?be.call(e.childNodes):null,a,!i&&t?t:o?o.__e:e.firstChild,i,r),Mi(a,n,r)}function zi(n,e){ee(n,e,zi)}function Yo(n,e,t){var i,o,a,r,l=Tn({},n.props);for(a in n.type&&n.type.defaultProps&&(r=n.type.defaultProps),e)a=="key"?i=e[a]:a=="ref"?o=e[a]:l[a]=e[a]===void 0&&r!==void 0?r[a]:e[a];return arguments.length>2&&(l.children=arguments.length>3?be.call(arguments,2):t),le(n.type,l,i||n.key,o||n.ref,null)}function $i(n){function e(t){var i,o;return this.getChildContext||(i=new Set,(o={})[e.__c]=this,this.getChildContext=function(){return o},this.componentWillUnmount=function(){i=null},this.shouldComponentUpdate=function(a){this.props.value!==a.value&&i.forEach(function(r){r.__e=!0,ft(r)})},this.sub=function(a){i.add(a);var r=a.componentWillUnmount;a.componentWillUnmount=function(){i&&i.delete(a),r&&r.call(a)}}),t.children}return e.__c="__cC"+Di++,e.__=n,e.Provider=e.__l=(e.Consumer=function(t,i){return t.children(i)}).contextType=e,e}be=Fi.slice,H={__e:function(n,e,t,i){for(var o,a,r;e=e.__;)if((o=e.__c)&&!o.__)try{if((a=o.constructor)&&a.getDerivedStateFromError!=null&&(o.setState(a.getDerivedStateFromError(n)),r=o.__d),o.componentDidCatch!=null&&(o.componentDidCatch(n,i||{}),r=o.__d),r)return o.__E=o}catch(l){n=l}throw n}},Pi=0,Cn.prototype.setState=function(n,e){var t;t=this.__s!=null&&this.__s!==this.state?this.__s:this.__s=Tn({},this.state),typeof n=="function"&&(n=n(Tn({},t),this.props)),n&&Tn(t,n),n!=null&&this.__v&&(e&&this._sb.push(e),ft(this))},Cn.prototype.forceUpdate=function(n){this.__v&&(this.__e=!0,n&&this.__h.push(n),ft(this))},Cn.prototype.render=E,Yn=[],Ei=typeof Promise=="function"?Promise.prototype.then.bind(Promise.resolve()):setTimeout,Li=function(n,e){return n.__v.__b-e.__v.__b},Ne.__r=0,Oi=/(PointerCapture)$|Capture$/i,kt=0,ut=Xt(!1),gt=Xt(!0),Di=0;var qo=0;function s(n,e,t,i,o,a){e||(e={});var r,l,c=e;if("ref"in c)for(l in c={},e)l=="ref"?r=e[l]:c[l]=e[l];var d={type:n,props:c,key:t,ref:r,__k:null,__:null,__b:0,__e:null,__c:null,constructor:void 0,__v:--qo,__i:-1,__u:0,__source:o,__self:a};if(typeof n=="function"&&(r=n.defaultProps))for(l in r)c[l]===void 0&&(c[l]=r[l]);return H.vnode&&H.vnode(d),d}var Un,X,Je,Jt,te=0,Vi=[],nn=H,ni=nn.__b,ei=nn.__r,ti=nn.diffed,ii=nn.__c,oi=nn.unmount,ai=nn.__;function ie(n,e){nn.__h&&nn.__h(X,n,te||e),te=0;var t=X.__H||(X.__H={__:[],__h:[]});return n>=t.__.length&&t.__.push({}),t.__[n]}function _n(n){return te=1,$e(ji,n)}function $e(n,e,t){var i=ie(Un++,2);if(i.t=n,!i.__c&&(i.__=[t?t(e):ji(void 0,e),function(l){var c=i.__N?i.__N[0]:i.__[0],d=i.t(c,l);c!==d&&(i.__N=[d,i.__[1]],i.__c.setState({}))}],i.__c=X,!X.__f)){var o=function(l,c,d){if(!i.__c.__H)return!0;var p=i.__c.__H.__.filter(function(f){return!!f.__c});if(p.every(function(f){return!f.__N}))return!a||a.call(this,l,c,d);var u=i.__c.props!==l;return p.forEach(function(f){if(f.__N){var g=f.__[0];f.__=f.__N,f.__N=void 0,g!==f.__[0]&&(u=!0)}}),a&&a.call(this,l,c,d)||u};X.__f=!0;var a=X.shouldComponentUpdate,r=X.componentWillUpdate;X.componentWillUpdate=function(l,c,d){if(this.__e){var p=a;a=void 0,o(l,c,d),a=p}r&&r.call(this,l,c,d)},X.shouldComponentUpdate=o}return i.__N||i.__}function W(n,e){var t=ie(Un++,3);!nn.__s&&Lt(t.__H,e)&&(t.__=n,t.u=e,X.__H.__h.push(t))}function me(n,e){var t=ie(Un++,4);!nn.__s&&Lt(t.__H,e)&&(t.__=n,t.u=e,X.__h.push(t))}function R(n){return te=5,$(function(){return{current:n}},[])}function Et(n,e,t){te=6,me(function(){if(typeof n=="function"){var i=n(e());return function(){n(null),i&&typeof i=="function"&&i()}}if(n)return n.current=e(),function(){return n.current=null}},t==null?t:t.concat(n))}function $(n,e){var t=ie(Un++,7);return Lt(t.__H,e)&&(t.__=n(),t.__H=e,t.__h=n),t.__}function ln(n,e){return te=8,$(function(){return n},e)}function xe(n){var e=X.context[n.__c],t=ie(Un++,9);return t.c=n,e?(t.__==null&&(t.__=!0,e.sub(X)),e.props.value):n.__}function Ui(n,e){nn.useDebugValue&&nn.useDebugValue(e?e(n):n)}function Gi(){var n=ie(Un++,11);if(!n.__){for(var e=X.__v;e!==null&&!e.__m&&e.__!==null;)e=e.__;var t=e.__m||(e.__m=[0,0]);n.__="P"+t[0]+"-"+t[1]++}return n.__}function Ko(){for(var n;n=Vi.shift();)if(n.__P&&n.__H)try{n.__H.__h.forEach(Ee),n.__H.__h.forEach(bt),n.__H.__h=[]}catch(e){n.__H.__h=[],nn.__e(e,n.__v)}}nn.__b=function(n){X=null,ni&&ni(n)},nn.__=function(n,e){n&&e.__k&&e.__k.__m&&(n.__m=e.__k.__m),ai&&ai(n,e)},nn.__r=function(n){ei&&ei(n),Un=0;var e=(X=n.__c).__H;e&&(Je===X?(e.__h=[],X.__h=[],e.__.forEach(function(t){t.__N&&(t.__=t.__N),t.u=t.__N=void 0})):(e.__h.forEach(Ee),e.__h.forEach(bt),e.__h=[],Un=0)),Je=X},nn.diffed=function(n){ti&&ti(n);var e=n.__c;e&&e.__H&&(e.__H.__h.length&&(Vi.push(e)!==1&&Jt===nn.requestAnimationFrame||((Jt=nn.requestAnimationFrame)||Qo)(Ko)),e.__H.__.forEach(function(t){t.u&&(t.__H=t.u),t.u=void 0})),Je=X=null},nn.__c=function(n,e){e.some(function(t){try{t.__h.forEach(Ee),t.__h=t.__h.filter(function(i){return!i.__||bt(i)})}catch(i){e.some(function(o){o.__h&&(o.__h=[])}),e=[],nn.__e(i,t.__v)}}),ii&&ii(n,e)},nn.unmount=function(n){oi&&oi(n);var e,t=n.__c;t&&t.__H&&(t.__H.__.forEach(function(i){try{Ee(i)}catch(o){e=o}}),t.__H=void 0,e&&nn.__e(e,t.__v))};var ri=typeof requestAnimationFrame=="function";function Qo(n){var e,t=function(){clearTimeout(i),ri&&cancelAnimationFrame(e),setTimeout(n)},i=setTimeout(t,100);ri&&(e=requestAnimationFrame(t))}function Ee(n){var e=X,t=n.__c;typeof t=="function"&&(n.__c=void 0,t()),X=e}function bt(n){var e=X;n.__c=n.__(),X=e}function Lt(n,e){return!n||n.length!==e.length||e.some(function(t,i){return t!==n[i]})}function ji(n,e){return typeof e=="function"?e(n):e}function Wi(n,e){for(var t in e)n[t]=e[t];return n}function ht(n,e){for(var t in n)if(t!=="__source"&&!(t in e))return!0;for(var i in e)if(i!=="__source"&&n[i]!==e[i])return!0;return!1}function Yi(n,e){var t=e(),i=_n({t:{__:t,u:e}}),o=i[0].t,a=i[1];return me(function(){o.__=t,o.u=e,nt(o)&&a({t:o})},[n,t,e]),W(function(){return nt(o)&&a({t:o}),n(function(){nt(o)&&a({t:o})})},[n]),t}function nt(n){var e,t,i=n.u,o=n.__;try{var a=i();return!((e=o)===(t=a)&&(e!==0||1/e==1/t)||e!=e&&t!=t)}catch{return!0}}function qi(n){n()}function Ki(n){return n}function Qi(){return[!1,qi]}var Zi=me;function mt(n,e){this.props=n,this.context=e}function Xi(n,e){function t(o){var a=this.props.ref,r=a==o.ref;return!r&&a&&(a.call?a(null):a.current=null),e?!e(this.props,o)||!r:ht(this.props,o)}function i(o){return this.shouldComponentUpdate=t,Mn(n,o)}return i.displayName="Memo("+(n.displayName||n.name)+")",i.prototype.isReactComponent=!0,i.__f=!0,i}(mt.prototype=new Cn).isPureReactComponent=!0,mt.prototype.shouldComponentUpdate=function(n,e){return ht(this.props,n)||ht(this.state,e)};var si=H.__b;H.__b=function(n){n.type&&n.type.__f&&n.ref&&(n.props.ref=n.ref,n.ref=null),si&&si(n)};var Zo=typeof Symbol<"u"&&Symbol.for&&Symbol.for("react.forward_ref")||3911;function Ji(n){function e(t){var i=Wi({},t);return delete i.ref,n(i,t.ref||null)}return e.$$typeof=Zo,e.render=e,e.prototype.isReactComponent=e.__f=!0,e.displayName="ForwardRef("+(n.displayName||n.name)+")",e}var li=function(n,e){return n==null?null:In(In(n).map(e))},Xo={map:li,forEach:li,count:function(n){return n?In(n).length:0},only:function(n){var e=In(n);if(e.length!==1)throw"Children.only";return e[0]},toArray:In},Jo=H.__e;H.__e=function(n,e,t,i){if(n.then){for(var o,a=e;a=a.__;)if((o=a.__c)&&o.__c)return e.__e==null&&(e.__e=t.__e,e.__k=t.__k),o.__c(n,e)}Jo(n,e,t,i)};var ci=H.unmount;function no(n,e,t){return n&&(n.__c&&n.__c.__H&&(n.__c.__H.__.forEach(function(i){typeof i.__c=="function"&&i.__c()}),n.__c.__H=null),(n=Wi({},n)).__c!=null&&(n.__c.__P===t&&(n.__c.__P=e),n.__c=null),n.__k=n.__k&&n.__k.map(function(i){return no(i,e,t)})),n}function eo(n,e,t){return n&&t&&(n.__v=null,n.__k=n.__k&&n.__k.map(function(i){return eo(i,e,t)}),n.__c&&n.__c.__P===e&&(n.__e&&t.appendChild(n.__e),n.__c.__e=!0,n.__c.__P=t)),n}function Le(){this.__u=0,this.o=null,this.__b=null}function to(n){var e=n.__.__c;return e&&e.__a&&e.__a(n)}function na(n){var e,t,i;function o(a){if(e||(e=n()).then(function(r){t=r.default||r},function(r){i=r}),i)throw i;if(!t)throw e;return Mn(t,a)}return o.displayName="Lazy",o.__f=!0,o}function se(){this.i=null,this.l=null}H.unmount=function(n){var e=n.__c;e&&e.__R&&e.__R(),e&&32&n.__u&&(n.type=null),ci&&ci(n)},(Le.prototype=new Cn).__c=function(n,e){var t=e.__c,i=this;i.o==null&&(i.o=[]),i.o.push(t);var o=to(i.__v),a=!1,r=function(){a||(a=!0,t.__R=null,o?o(l):l())};t.__R=r;var l=function(){if(!--i.__u){if(i.state.__a){var c=i.state.__a;i.__v.__k[0]=eo(c,c.__c.__P,c.__c.__O)}var d;for(i.setState({__a:i.__b=null});d=i.o.pop();)d.forceUpdate()}};i.__u++||32&e.__u||i.setState({__a:i.__b=i.__v.__k[0]}),n.then(r,r)},Le.prototype.componentWillUnmount=function(){this.o=[]},Le.prototype.render=function(n,e){if(this.__b){if(this.__v.__k){var t=document.createElement("div"),i=this.__v.__k[0].__c;this.__v.__k[0]=no(this.__b,t,i.__O=i.__P)}this.__b=null}var o=e.__a&&Mn(E,null,n.fallback);return o&&(o.__u&=-33),[Mn(E,null,e.__a?null:n.children),o]};var pi=function(n,e,t){if(++t[1]===t[0]&&n.l.delete(e),n.props.revealOrder&&(n.props.revealOrder[0]!=="t"||!n.l.size))for(t=n.i;t;){for(;t.length>3;)t.pop()();if(t[1]<t[0])break;n.i=t=t[2]}};function ea(n){return this.getChildContext=function(){return n.context},n.children}function ta(n){var e=this,t=n.h;e.componentWillUnmount=function(){ee(null,e.v),e.v=null,e.h=null},e.h&&e.h!==t&&e.componentWillUnmount(),e.v||(e.h=t,e.v={nodeType:1,parentNode:t,childNodes:[],contains:function(){return!0},appendChild:function(i){this.childNodes.push(i),e.h.appendChild(i)},insertBefore:function(i,o){this.childNodes.push(i),e.h.insertBefore(i,o)},removeChild:function(i){this.childNodes.splice(this.childNodes.indexOf(i)>>>1,1),e.h.removeChild(i)}}),ee(Mn(ea,{context:e.context},n.__v),e.v)}function ia(n,e){var t=Mn(ta,{__v:n,h:e});return t.containerInfo=e,t}(se.prototype=new Cn).__a=function(n){var e=this,t=to(e.__v),i=e.l.get(n);return i[0]++,function(o){var a=function(){e.props.revealOrder?(i.push(o),pi(e,n,i)):o()};t?t(a):a()}},se.prototype.render=function(n){this.i=null,this.l=new Map;var e=In(n.children);n.revealOrder&&n.revealOrder[0]==="b"&&e.reverse();for(var t=e.length;t--;)this.l.set(e[t],this.i=[1,0,this.i]);return n.children},se.prototype.componentDidUpdate=se.prototype.componentDidMount=function(){var n=this;this.l.forEach(function(e,t){pi(n,t,e)})};var io=typeof Symbol<"u"&&Symbol.for&&Symbol.for("react.element")||60103,oa=/^(?:accent|alignment|arabic|baseline|cap|clip(?!PathU)|color|dominant|fill|flood|font|glyph(?!R)|horiz|image(!S)|letter|lighting|marker(?!H|W|U)|overline|paint|pointer|shape|stop|strikethrough|stroke|text(?!L)|transform|underline|unicode|units|v|vector|vert|word|writing|x(?!C))[A-Z]/,aa=/^on(Ani|Tra|Tou|BeforeInp|Compo)/,ra=/[A-Z0-9]/g,sa=typeof document<"u",la=function(n){return(typeof Symbol<"u"&&typeof Symbol()=="symbol"?/fil|che|rad/:/fil|che|ra/).test(n)};function ca(n,e,t){return e.__k==null&&(e.textContent=""),ee(n,e),typeof t=="function"&&t(),n?n.__c:null}function pa(n,e,t){return zi(n,e),typeof t=="function"&&t(),n?n.__c:null}Cn.prototype.isReactComponent={},["componentWillMount","componentWillReceiveProps","componentWillUpdate"].forEach(function(n){Object.defineProperty(Cn.prototype,n,{configurable:!0,get:function(){return this["UNSAFE_"+n]},set:function(e){Object.defineProperty(this,n,{configurable:!0,writable:!0,value:e})}})});var di=H.event;function da(){}function ua(){return this.cancelBubble}function ga(){return this.defaultPrevented}H.event=function(n){return di&&(n=di(n)),n.persist=da,n.isPropagationStopped=ua,n.isDefaultPrevented=ga,n.nativeEvent=n};var Ot,fa={enumerable:!1,configurable:!0,get:function(){return this.class}},ui=H.vnode;H.vnode=function(n){typeof n.type=="string"&&function(e){var t=e.props,i=e.type,o={},a=i.indexOf("-")===-1;for(var r in t){var l=t[r];if(!(r==="value"&&"defaultValue"in t&&l==null||sa&&r==="children"&&i==="noscript"||r==="class"||r==="className")){var c=r.toLowerCase();r==="defaultValue"&&"value"in t&&t.value==null?r="value":r==="download"&&l===!0?l="":c==="translate"&&l==="no"?l=!1:c[0]==="o"&&c[1]==="n"?c==="ondoubleclick"?r="ondblclick":c!=="onchange"||i!=="input"&&i!=="textarea"||la(t.type)?c==="onfocus"?r="onfocusin":c==="onblur"?r="onfocusout":aa.test(r)&&(r=c):c=r="oninput":a&&oa.test(r)?r=r.replace(ra,"-$&").toLowerCase():l===null&&(l=void 0),c==="oninput"&&o[r=c]&&(r="oninputCapture"),o[r]=l}}i=="select"&&o.multiple&&Array.isArray(o.value)&&(o.value=In(t.children).forEach(function(d){d.props.selected=o.value.indexOf(d.props.value)!=-1})),i=="select"&&o.defaultValue!=null&&(o.value=In(t.children).forEach(function(d){d.props.selected=o.multiple?o.defaultValue.indexOf(d.props.value)!=-1:o.defaultValue==d.props.value})),t.class&&!t.className?(o.class=t.class,Object.defineProperty(o,"className",fa)):(t.className&&!t.class||t.class&&t.className)&&(o.class=o.className=t.className),e.props=o}(n),n.$$typeof=io,ui&&ui(n)};var gi=H.__r;H.__r=function(n){gi&&gi(n),Ot=n.__c};var fi=H.diffed;H.diffed=function(n){fi&&fi(n);var e=n.props,t=n.__e;t!=null&&n.type==="textarea"&&"value"in e&&e.value!==t.value&&(t.value=e.value==null?"":e.value),Ot=null};var ba={ReactCurrentDispatcher:{current:{readContext:function(n){return Ot.__n[n.__c].props.value},useCallback:ln,useContext:xe,useDebugValue:Ui,useDeferredValue:Ki,useEffect:W,useId:Gi,useImperativeHandle:Et,useInsertionEffect:Zi,useLayoutEffect:me,useMemo:$,useReducer:$e,useRef:R,useState:_n,useSyncExternalStore:Yi,useTransition:Qi}}};function ha(n){return Mn.bind(null,n)}function Ve(n){return!!n&&n.$$typeof===io}function ma(n){return Ve(n)&&n.type===E}function xa(n){return!!n&&!!n.displayName&&(typeof n.displayName=="string"||n.displayName instanceof String)&&n.displayName.startsWith("Memo(")}function _a(n){return Ve(n)?Yo.apply(null,arguments):n}function ya(n){return!!n.__k&&(ee(null,n),!0)}function wa(n){return n&&(n.base||n.nodeType===1&&n)||null}var va=function(n,e){return n(e)},Ca=function(n,e){return n(e)},Sa=E,ka=Ve,Aa={useState:_n,useId:Gi,useReducer:$e,useEffect:W,useLayoutEffect:me,useInsertionEffect:Zi,useTransition:Qi,useDeferredValue:Ki,useSyncExternalStore:Yi,startTransition:qi,useRef:R,useImperativeHandle:Et,useMemo:$,useCallback:ln,useContext:xe,useDebugValue:Ui,version:"18.3.1",Children:Xo,render:ca,hydrate:pa,unmountComponentAtNode:ya,createPortal:ia,createElement:Mn,createContext:$i,createFactory:ha,cloneElement:_a,createRef:pe,Fragment:E,isValidElement:Ve,isElement:ka,isFragment:ma,isMemo:xa,findDOMNode:wa,Component:Cn,PureComponent:mt,memo:Xi,forwardRef:Ji,flushSync:Ca,unstable_batchedUpdates:va,StrictMode:Sa,Suspense:Le,SuspenseList:se,lazy:na,__SECRET_INTERNALS_DO_NOT_USE_OR_YOU_WILL_BE_FIRED:ba};function _e(n){const{name:e,initialState:t,reducers:i}=n,o=(r=t,l)=>{const c=l.type.replace(`${e}/`,""),d=i[c];return d?l.payload!==void 0?d(r,l.payload):d(r):r},a=Object.keys(i).reduce((r,l)=>(r[l]=c=>c!==void 0?{type:`${e}/${l}`,payload:c}:{type:`${e}/${l}`},r),{});return{name:e,initialState:t,reducer:o,actions:a}}const Ta={currentView:"",previousView:"",startingView:"",advRootProps:{},proRootProps:{},isMobile:!1,shouldShowDisabledSubmitMessage:!1,effectiveLocale:"en",deadCmTimeoutId:null,partialLoadTimeoutId:null,isFullyLoaded:!1},Pa={initViews:(n,e)=>({...n,currentView:e,previousView:e,startingView:e}),setCurrentView:(n,e)=>({...n,previousView:n.currentView,currentView:e}),setAdvRootProps:(n,e)=>({...n,advRootProps:e}),setProRootProps:(n,e)=>({...n,proRootProps:e}),setEffectiveLocale:(n,e)=>({...n,effectiveLocale:e}),setIsMobile:(n,e)=>({...n,isMobile:e}),showDisableMessage:n=>n.shouldShowDisabledSubmitMessage?n:{...n,shouldShowDisabledSubmitMessage:!0},hideDisableMessage:n=>n.shouldShowDisabledSubmitMessage?{...n,shouldShowDisabledSubmitMessage:!1}:n,setDeadCmTimeout:(n,e)=>({...n,deadCmTimeoutId:e}),clearDeadCmTimeout:n=>(n.deadCmTimeoutId!==null&&clearTimeout(n.deadCmTimeoutId),{...n,deadCmTimeoutId:null}),setPartialLoadTimeout:(n,e)=>({...n,partialLoadTimeoutId:e}),clearPartialLoadTimeout:n=>(n.partialLoadTimeoutId!==null&&clearTimeout(n.partialLoadTimeoutId),{...n,partialLoadTimeoutId:null}),markAsFullyLoaded:n=>({...n,isFullyLoaded:!0})},Dt=_e({name:"globalState",initialState:Ta,reducers:Pa}),{initViews:Ea,setCurrentView:un,setAdvRootProps:La,setProRootProps:Vc,setIsMobile:Oa,setEffectiveLocale:oo,showDisableMessage:et,hideDisableMessage:tt,setDeadCmTimeout:Da,clearDeadCmTimeout:Ie,setPartialLoadTimeout:Fa,clearPartialLoadTimeout:xt,markAsFullyLoaded:Na}=Dt.actions,Ia={config:{defaultCategory:"",privacyPolicy:"",languages:[],defaultOptin:"CUSTOM",defaultCategoriesMap:{},secondStepOptIn:!1,experienceType:"LEGACY",iabCombination:"",gppCountryStates:[]},categories:[],categoriesWithIab:[],contents:{css_key:""},sessionId:""},Ra={config:Ia,iabData:null,gppData:null,loaded:!1,loading:!1,error:null,iabLoading:!1,iabError:null,gppLoading:!1,gppError:null,vendorDisclosures:{},vendorDisclosuresLoading:{},vendorDisclosuresError:{},vendorDisclosuresFetchedAt:{}},Ma={fetchConfigStart:n=>({...n,loaded:!1,loading:!0,error:null}),fetchConfigSuccess:(n,e)=>{const t={code:"en",label:"English"},i=e.config.languages&&e.config.languages.some(o=>o.code==="en")?e.config.languages:[t,...e.config.languages||[]];return{...n,config:{...e,config:{...e.config,languages:i}},loaded:!0,loading:!1,error:null}},fetchConfigFailure:(n,e)=>({...n,loading:!1,error:e}),fetchIabDataStart:n=>({...n,iabLoading:!0,iabError:null}),fetchIabDataSuccess:(n,e)=>({...n,iabData:e,iabLoading:!1,iabError:null}),fetchIabDataFailure:(n,e)=>({...n,iabLoading:!1,iabError:e}),fetchVendorDisclosuresStart:(n,e)=>({...n,vendorDisclosuresLoading:{...n.vendorDisclosuresLoading,[e]:!0}}),fetchVendorDisclosuresSuccess:(n,e)=>({...n,vendorDisclosures:{...n.vendorDisclosures,[e.vendorId]:e.disclosures},vendorDisclosuresLoading:{...n.vendorDisclosuresLoading,[e.vendorId]:!1},vendorDisclosuresError:{...n.vendorDisclosuresError,[e.vendorId]:!1},vendorDisclosuresFetchedAt:{...n.vendorDisclosuresFetchedAt,[e.vendorId]:Date.now()}}),fetchVendorDisclosuresFailure:(n,e)=>({...n,vendorDisclosuresLoading:{...n.vendorDisclosuresLoading,[e]:!1},vendorDisclosuresError:{...n.vendorDisclosuresError,[e]:!0}}),fetchGppDataStart:n=>({...n,gppLoading:!0,gppError:null}),fetchGppDataSuccess:(n,e)=>({...n,gppData:e,gppLoading:!1,gppError:null}),fetchGppDataFailure:(n,e)=>({...n,gppLoading:!1,gppError:e})},Jn=_e({name:"config",initialState:Ra,reducers:Ma}),Ba={headerDescriptionElements:[],titleLogoExist:!1},Ha={addHeaderDescElem:(n,e)=>({...n,headerDescriptionElements:[...n.headerDescriptionElements,e]}),clearHeaderDescElems:n=>({...n,headerDescriptionElements:[]}),setTitleLogoExist:(n,e)=>({...n,titleLogoExist:e})},Ft=_e({name:"uiState",initialState:Ba,reducers:Ha}),{addHeaderDescElem:za,setTitleLogoExist:$a,clearHeaderDescElems:Va}=Ft.actions,Ua=["twoStepPanel","loyaltyMessage"];function Ga(){return!1}function fn(n){return(n==null?void 0:n.layout)==="gpp"}function Bn(n){return(n==null?void 0:n.layout)==="gdpr"||(n==null?void 0:n.layout)==="gdpr_accessible"}function Pn(n){return(n==null?void 0:n.layout)==="default_eu"||(n==null?void 0:n.layout)==="default_eu_accessible"}function Nn(n){return(n==null?void 0:n.layout)==="default_eu_accessible"}function ja(n){return(n==null?void 0:n.layout)==="gdpr_accessible"}function j(n){var e;return((e=n==null?void 0:n.layout)==null?void 0:e.endsWith("_accessible"))||(n==null?void 0:n.layout)==="gpp"}function Nt(n){return Ua.includes(n)}function It(n,e){return n.cmType==="ADV"?!!j(n):n.cmType==="PRO"&&e?e.experienceType==="WCAG2":!1}const _t={current:null},Wa=n=>{_t.current=n},Rt=()=>_t.current?_t.current:null,Ya=()=>{const n=Rt();if(n)return n.globalStateSlice.advRootProps.debug||n.globalStateSlice.proRootProps.debug},qa=()=>{const n=Rt();if(n)return n.globalStateSlice.advRootProps.preview},oe={setAppState:Wa,getAppState:Rt,inDebugMode:Ya,isPreview:qa},Ka={isLoading:!1,error:null},Qa={logSendStarted:n=>({...n,loading:!0,error:null}),logSendSuccess:n=>({...n,loading:!1,error:null}),logSendFailed:(n,e)=>({...n,loading:!1,error:e})},ye=_e({name:"activityLogging",initialState:Ka,reducers:Qa});function Mt(n,...e){const t=n.replace(/\{}/g,()=>{const i=e.shift();return i!==void 0?String(i):"{}"});return e.length>0?`${t} ${e.join(" ")}`:t}let bi;const Za=()=>(bi??(bi=oe.inDebugMode()),bi),Xa=(n,...e)=>{if(!Za())return;const t=Mt(n,...e);console.log(t)},Ja=(n,...e)=>{const t=Mt(n,...e);console.warn(t)},nr=(n,...e)=>{const t=Mt(n,...e);console.error(t)},T={info:Xa,warn:Ja,error:nr},er={OPT_OUT:1},Wn={SHOW_YES:!0,SHOW_NO:!1,UNSET:null},re={CALIFORNIA:"ca",COLORADO:"co",CONNECTICUT:"ct",VIRGINIA:"va",UTAH:"ut"},tr={SaleOptOut:"Sale of Personal Information",SharingOptOut:"Sharing of Personal Information",TargetedAdvertisingOptOut:"Targeted Advertising",SensitiveDataProcessing:"Sensitive Data Processing",KnownChildSensitiveDataConsents:"Known Child Consents Status",PersonalDataConsents:"Personal Data Consent"},ir={racialEthnic:"Racial, Ethnic",religiousBelief:"Religious Belief",health:"Health",sexualOrientation:"Sexual Orientation",immigration:"Immigration",geneticData:"Genetic Data",biometric:"Biometric",geolocation:"Geolocation",sensitivePersonalInfo:"Sensitive Personal Information",sensitiveSecurityInfo:"Sensitive Security Information",unionMembership:"Union Membership",communication:"Communications",racialEthnicReligiousBeliefUnion:"Racial or Ethnic Origin, Religious or Philosophical Beliefs, or Union Membership",consentAge13To16:"Consent for the Personal Data from Age 13-16",consentAgeYoungerThan13:"Consent for Personal Data younger than 13 Years of Age",sellLessThan16:"Consent to Sell the Personal Information of Consumers Less Than 16 years of Age",shareLessThan16:"Consent to Share the Personal Information of Consumers Less Than 16 years of Age",sensitiveDataKnownChild:"Consent to Process Sensitive Data from a Known Child",sellData13To16:"Consent to Sell the Personal Data of Consumers from Age 13-16",targetedAdvertising13To16:"Consent to Process the Personal Data of Consumers from Age 13-16 for Purposes of Targeted Advertising"};function hi(n){return tr[n]||n}function or(n){return ir[n]||n}function mi(n){return n.subsections&&n.subsections.length>0}function ar(n){var e;return((e=n.subsections)==null?void 0:e.filter(t=>t.enabled))||[]}function rr(n){return`gpp-purpose-${n.id}`}function sr(n){return`gpp-subsection-${n.gppPurposeId}-${n.name}`}function yt(n){switch(n==null?void 0:n.toLowerCase()){case re.CALIFORNIA:return{sensitiveData:["sensitivePersonalInfo","sensitiveSecurityInfo","geolocation","racialEthnicReligiousBeliefUnion","communication","geneticData","biometric","health","sexualOrientation"],knownChild:["sellLessThan16","shareLessThan16"]};case re.COLORADO:return{sensitiveData:["racialEthnic","religiousBelief","health","sexualOrientation","immigration","geneticData","biometric"],knownChild:[]};case re.CONNECTICUT:return{sensitiveData:["racialEthnic","religiousBelief","health","sexualOrientation","immigration","geneticData","biometric","geolocation"],knownChild:["sensitiveDataKnownChild","sellData13To16","targetedAdvertising13To16"]};case re.UTAH:return{sensitiveData:["racialEthnic","religiousBelief","health","sexualOrientation","immigration","geneticData","biometric","geolocation"],knownChild:[]};case re.VIRGINIA:return{sensitiveData:["racialEthnic","religiousBelief","health","sexualOrientation","immigration","geneticData","biometric","geolocation"],knownChild:[]};default:return{sensitiveData:["racialEthnic","religiousBelief","health","sexualOrientation","immigration","geneticData","biometric","geolocation","sensitivePersonalInfo","sensitiveSecurityInfo","unionMembership","communication"],knownChild:["consentAge13To16","consentAgeYoungerThan13"]}}}const dn={NOT_APPLICABLE:0,OPTED_OUT:1,OPTED_IN:2},Re={NOT_APPLICABLE:-2,NO_CATEGORY_MAPPING:-1},Y={NOT_APPLICABLE:0,YES:1,NO:2},mn={SALE_OPT_OUT_NOTICE:"SaleOptOutNotice",SALE_OPT_OUT:"SaleOptOut",SENSITIVE_DATA_PROCESSING:"SensitiveDataProcessing",TARGETED_ADVERTISING_OPT_OUT:"TargetedAdvertisingOptOut",SHARING_NOTICE:"SharingNotice",KNOWN_CHILD_SENSITIVE_DATA_CONSENTS:"KnownChildSensitiveDataConsents",TARGETED_ADVERTISING_OPT_OUT_NOTICE:"TargetedAdvertisingOptOutNotice",PERSONAL_DATA_CONSENTS:"PersonalDataConsents",MSPA_COVERED_TRANSACTION:"MspaCoveredTransaction",MSPA_OPT_OUT_OPTION_MODE:"MspaOptOutOptionMode",MSPA_SERVICE_PROVIDER_MODE:"MspaServiceProviderMode",SHARING_OPT_OUT:"SharingOptOut"},ao=n=>n===!0?dn.OPTED_IN:n===!1?dn.OPTED_OUT:dn.NOT_APPLICABLE,xi=n=>n===dn.OPTED_IN,lr=(n,e=1)=>{n.SaleOptOut!=null&&(n.SaleOptOutNotice=n.SaleOptOut===dn.NOT_APPLICABLE?dn.NOT_APPLICABLE:e),n.MspaCoveredTransaction=Y.YES,n.MspaOptOutOptionMode=Y.NO,n.MspaServiceProviderMode=Y.YES},_i=(n,e,t)=>t.map(i=>{const o=`${e}_${i}`,a=n[o];return ao(a)}),cr=(n,e)=>{const t={};return Object.entries(n.activeGppPurposes).forEach(([i,o])=>{const a=ao(o);switch(i){case mn.SALE_OPT_OUT_NOTICE:t.SaleOptOutNotice=a;break;case mn.SALE_OPT_OUT:t.SaleOptOut=a;break;case mn.TARGETED_ADVERTISING_OPT_OUT:t.TargetedAdvertisingOptOut=a;break;case mn.SHARING_NOTICE:t.SharingNotice=a;break;case mn.KNOWN_CHILD_SENSITIVE_DATA_CONSENTS:if(e){const l=yt(e.stateCode).knownChild;l.length>0?t.KnownChildSensitiveDataConsents=_i(n.activeGppSubsections,i,l):t.KnownChildSensitiveDataConsents=a}else t.KnownChildSensitiveDataConsents=a;break;case mn.TARGETED_ADVERTISING_OPT_OUT_NOTICE:t.TargetedAdvertisingOptOutNotice=a;break;case mn.PERSONAL_DATA_CONSENTS:t.PersonalDataConsents=a;break;case mn.MSPA_COVERED_TRANSACTION:t.MspaCoveredTransaction=a;break;case mn.MSPA_OPT_OUT_OPTION_MODE:t.MspaOptOutOptionMode=a;break;case mn.MSPA_SERVICE_PROVIDER_MODE:t.MspaServiceProviderMode=a;break;case mn.SHARING_OPT_OUT:t.SharingOptOut=a;break;case mn.SENSITIVE_DATA_PROCESSING:if(e){const l=yt(e.stateCode).sensitiveData;t.SensitiveDataProcessing=_i(n.activeGppSubsections,i,l)}break}}),lr(t),t},pr=(n,e)=>{var o;const t=cr(n,e);switch(((o=e==null?void 0:e.stateCode)==null?void 0:o.toLowerCase())||"nat"){case"ca":{const a={...t};a.SharingOptOutNotice=a.SharingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES;const r=a.SensitiveDataProcessing;let l=!0;if(r){for(const c of r)if(c!==dn.NOT_APPLICABLE){l=!1;break}}return a.SensitiveDataLimitUseNotice=l?Y.NOT_APPLICABLE:Y.YES,a}case"va":{const a={...t};return a.SharingNotice=Y.YES,a.TargetedAdvertisingOptOutNotice=a.TargetedAdvertisingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES,a}case"ct":{const a={...t};return a.SharingNotice=Y.YES,a.TargetedAdvertisingOptOutNotice=a.TargetedAdvertisingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES,a}case"co":{const a={...t};return a.SharingNotice=Y.YES,a.TargetedAdvertisingOptOutNotice=a.TargetedAdvertisingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES,a}case"ut":{const a={...t};return a.SharingNotice=Y.YES,a.TargetedAdvertisingOptOutNotice=a.TargetedAdvertisingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES,a.SensitiveDataProcessingOptOutNotice=Y.YES,a}default:{const a={...t};a.SharingNotice=Y.YES,a.SharingOptOutNotice=a.SharingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES,a.TargetedAdvertisingOptOutNotice=a.TargetedAdvertisingOptOut===dn.NOT_APPLICABLE?Y.NOT_APPLICABLE:Y.YES,a.SensitiveDataProcessingOptOutNotice=Y.YES;const r=a.SensitiveDataProcessing;let l=!0;if(r){for(const c of r)if(c!==dn.NOT_APPLICABLE){l=!1;break}}return a.SensitiveDataLimitUseNotice=l?Y.NOT_APPLICABLE:Y.YES,a}}},Bt=n=>{const{MspaCoveredTransaction:e,MspaOptOutOptionMode:t,MspaServiceProviderMode:i,...o}=n;return JSON.stringify(o)};var yn=(n=>(n.ACCEPT_ALL="ACCEPT_ALL",n.AGREE_AND_PROCEED="AGREE_AND_PROCEED",n.DECLINE_ALL="DECLINE_ALL",n.SUBMIT_PREFERENCES="OPT_IN",n.REQUIRED_ONLY="REQUIRED_ONLY",n))(yn||{}),wt=(n=>(n.OPT_IN_ALL="OPT_IN_ALL",n.OPT_OUT_ALL="OPT_OUT_ALL",n))(wt||{}),Me=(n=>(n.ASK_ME_LATER="ASK_ME_LATER",n.MORE_INFORMATION="MORE_INFORMATION",n.TWO_STEP_OPT_IN="TWO_STEP_OPT_IN",n))(Me||{});const dr={AGREE_AND_PROCEED:9,ACCEPT_ALL:10,DECLINE_ALL:11,REQUIRED_ONLY:12,MORE_INFORMATION:13,ASK_ME_LATER:14,TWO_STEP_OPT_IN:17,OPT_IN:18,OPT_IN_ALL:19,OPT_OUT_ALL:20};var ro=(n=>(n.ENDPOINT_SEND_LOG_CONSENT="consent/log",n))(ro||{});function Ht(n){if(typeof n=="string")return n;try{const e=new Set;return JSON.stringify(n,(i,o)=>{if(o&&typeof o=="object"){if(e.has(o))return"[Circular]";e.add(o)}return o})}catch(e){return T.error("Error in jsonStringify object:",e),String(n)}}function so(n,e){const t=new Array(e.length);return n.forEach(i=>{const o=i.value;o>=0&&o<e.length?(t[o]||(t[o]=[]),t[o].push(i.name)):o===Re.NOT_APPLICABLE?T.info("GPP purpose {} not applicable",i.name):o===Re.NO_CATEGORY_MAPPING?T.info("GPP purpose {} no category mapping",i.name):T.info("GPP purpose {} invalid category index {}",i.name,o)}),t}function ur(n,e,t,i,o){const a=[0];if(T.info("GPP submit preference - gppShowCategories: {}",o),o)for(let r=1;r<t.length;r++){const l=t[r],c=i[l.id];l&&c===!0&&a.push(r)}else{const r=so(n,t);for(let l=1;l<r.length;l++){const c=r[l];c&&c.length>0&&(c.every(p=>e[p]===!0)?(a.push(l),T.info("Added category index {} to preference list (all {} purposes consented)",l,c.length)):T.info("Skipped category index {} - not all purposes consented: {}",l,c.map(p=>`${p}: ${e[p]}`).join(", ")))}}return T.info("GPP submit preference list: {}",a),a}function gr(n,e,t){const i=[0];if(t)for(let o=1;o<e.length;o++)i.push(o);else{const o=so(n,e);for(let a=1;a<o.length;a++){const r=o[a];r&&r.length>0&&i.push(a)}}return T.info("GPP accept all list: {}",i),i}var Kn=(n=>(n.REMOVE_IFRAME="remove_iframe",n.TWO_STEP_NO="two_step_no",n.TWO_STEP_YES="two_step_yes",n.FINPROG_NO="finprog_no",n.FINPROG_YES="finprog_yes",n.CHANGE_PANEL="change_panel",n.SUBMIT_PREFERENCES="submit_preferences",n.PROCESS_IAB_VENDOR_LIST="process_iab_vendor_list",n.PROCESS_IAB_ACCEPT_ALL="process_iab_accept_all",n.PROCESS_IAB_REJECT_ALL="process_iab_reject_all",n.PROCESS_GPP_CONSENT="process_gpp_consent",n.PROCESS_GPP_ACCEPT_ALL="process_gpp_accept_all",n.PROCESS_GPP_REJECT_ALL="process_gpp_reject_all",n.REQUEST_IAB_GPP_CONSENT_DATA="request_iab_gpp_consent_data",n.CREATE_COOKIE="create_cookie",n))(Kn||{});function fr({messageType:n,data:e}){return{source:"preference_manager",message:n,data:Ht(e)}}function br({messageType:n,data:e}){return Ht({source:"preference_manager",message:n,data:e})}const hr="truste.eu.shouldShowCloseButton";function lo(n){return n.split(".").reduce((e,t)=>e==null?void 0:e[t],window)}function we(n){const e=lo(n);return e==null?(oe.isPreview()||T.error(`'${n}' does not exist`),!1):!0}function Qn(n){return typeof lo(n)!="function"?(oe.isPreview()||T.error(`'${n}' is not a function`),!1):!0}function bn(){return window.truste}function mr(){return bn().eu.bindMap}function En(){Gn({messageType:"remove_iframe",data:"true"})}function Be(n){if(!Qn("truste.eu.actmessage"))return;const t=fr(n);bn().eu.actmessage(t)}function Gn(n){window.postMessage(br(n),window.location.origin)}function de(n,e){Qn("truste.util.fireCustomEvent")&&(T.info("fireCustomEvent: {}",n),bn().util.fireCustomEvent(n,e))}function xr(n,e){_r(n),yr(n);const{activeCategories:t,activeVendors:i}=n;e(ho({activeCategories:t,activeVendors:i}))}function _r(n){const{activeCategories:e,allCategories:t,cmType:i,contents:o}=n,a=Vn(i,o);let r=jt(e,t,a);if(i==="PRO"){const l=Wt(t).map(({originalIndex:c})=>c);r=r.filter(c=>l.includes(c))}window.PREF_MGR_API_DEBUG&&window.PREF_MGR_API_DEBUG.authorities.push(window.location.hostname),Gn({messageType:"submit_preferences",data:Gt(r,Vn(i,o)&&r.length===0)}),T.info("saveOptInsToLocalStorageAndAppStore: {}",r)}function yr(n){const e=Qn("truste.util.setStorage"),t=we("truste.eu.COOKIE_CATEGORY_NAME");if(!e||!t)return;const i=wr(n);bn().util.setStorage(bn().eu.COOKIE_CATEGORY_NAME,i,!1)}function wr(n){const{allCategories:e,activeVendors:t,activeCategories:i,cmType:o,contents:a}=n,r={};return e.forEach((l,c)=>{const d={};l.vendors.forEach(p=>{p.domains&&p.domains.forEach(u=>{let f="0";if(c===0&&!Vn(o,a))f="2";else{const g=xn(l.id,p.id);t[g]!==void 0?f=t[g]?"1":"0":f=i!=null&&i[l.id]?"1":"0"}d[u]=f})}),r[l.name]={value:`${c}`,domains:d}}),r.version="1",r}function zt(){var t,i,o,a;const n=(o=(i=(t=bn())==null?void 0:t.eu)==null?void 0:i.bindMap)==null?void 0:o.prefCookie;if(n==null)return;const e=(a=bn())==null?void 0:a.util;return(typeof(e==null?void 0:e.getShiftCookie)=="function"?e.getShiftCookie(n,!1):void 0)??n}function vr(n){const{cmConfig:e,cmType:t,dispatch:i}=n;if(!we("truste.eu.bindMap"))return;const a=zt(),r=a&&!a.includes("-1"),l=a===""&&Vn(t,e.contents);if(r||l){const c=Ar();Sr(a,c,e.categories,i)}else kr(n)}function Cr(){return zt()!=null}function co(){const n=zt();return n!=null&&!n.includes("-1")}function Sr(n,e,t,i){T.info("Restore Preference State from bindMap->prefCookie and local storage: {}",n);let o,a;n===""?(o=[],a=0):(o=n.split(/[,|]/).map(c=>parseInt(c.trim(),10)),a=Math.max(...o)),i(fo({level:a,allCategories:t}));const r={};t.forEach((c,d)=>{r[c.id]=o.includes(d)});const l={};if(e&&Object.keys(e).length>0)for(const[c,d]of Object.entries(e)){if(c==="version"&&d===Ur)continue;const{domains:p,value:u}=d,f=t.find(x=>x.name===c);if(!f)continue;const g=f.vendors??[];for(const x of g){const b=x.domains.map(m=>p[m]).filter(m=>m!==void 0),h=u==="0"?!0:b.some(m=>m!=="0");l[xn(f.id,x.id)]=h}}i(bo({activeCategories:r,activeVendors:l,allCategories:t})),i(ho({activeCategories:r,activeVendors:l}))}function kr(n){const{cmConfig:e,cmType:t,dispatch:i}=n;if(e.categories.length===0)return;const o={};if(!Vn(t,e.contents)&&(o[e.categories[0].id]=!0,e.categories.length<=1))return;function a(){let c=e.config.defaultCategory==="deselect"?null:!0;for(let d=0;d<e.categories.length;d++){const p=e.categories[d];o[p.id]=d===0?!Vn(t,e.contents):c,p.id===e.config.defaultCategory&&e.config.defaultCategory!=="deselect"&&(c=!1)}}function r(){const c=Vn(t,e.contents)?0:1;if(e.config.defaultOptin==="CUSTOM")for(let d=c;d<e.categories.length;d++){const p=e.categories[d].id,u=e.config.defaultCategoriesMap[p];o[p]=u==="YES"?!0:u==="DESELECT"?null:!1}else{let d;e.config.defaultOptin==="ALL_YES"?d=!0:e.config.defaultOptin==="ALL_NO"?d=!1:e.config.defaultOptin==="ALL_DESELECTED"?d=null:d=!0;for(let p=c;p<e.categories.length;p++){const u=e.categories[p];o[u.id]=d}}}t==="ADV"?a():r();const l={};e.categories.forEach(c=>{const d=o[c.id];c.vendors.forEach(p=>{l[xn(c.id,p.id)]=d})}),i(bo({activeCategories:o,activeVendors:l,allCategories:e.categories}))}function Ar(){const n=Qn("truste.util.getStorage"),e=we("truste.eu.COOKIE_CATEGORY_NAME");return!n||!e?{}:bn().util.getStorage(bn().eu.COOKIE_CATEGORY_NAME)}function $t(n){return we("truste.eu.bindMap."+n)?bn().eu.bindMap[n]??"":""}function ve(n){var t,i,o;const e=(o=(i=(t=bn())==null?void 0:t.eu)==null?void 0:i.bindMap)==null?void 0:o.feat;return n.includes(".")?!!n.split(".").reduce((a,r)=>a==null?void 0:a[r],e):!!(e!=null&&e[n])}function Tr(){var e,t,i;return we("truste.eu.bindMap.hideCloseButtonEnabled")?!!((i=(t=(e=bn())==null?void 0:e.eu)==null?void 0:t.bindMap)!=null&&i.hideCloseButtonEnabled):!1}function Vt(){return $t("behavior")}let it="";function hn(n){if(!it){const e=(n==null?void 0:n.cmType)==="ADV"&&n.iconBaseUrl;let t;e?t=e:t=$t("iconBaseUrl"),t.trim()!==""&&(it=t.endsWith("/")?t:t+"/")}return it}async function Pr(n,e){if(oe.isPreview()){T.info("[ActivityLog] Skipping sendAnalytics in preview mode: action={}, categories={}",n,e);return}if(!Qn("truste.util.sendAnalytics"))return;const i=dr[n];if(i===void 0)throw new Error(`Unrecognized action type: ${n}`);await bn().util.sendAnalytics(i,e)}function Er(n,e){const t=pr(n,e),i=Bt(t);T.info("sendGppConsentMessage: {}",t),Gn({messageType:"process_gpp_consent",data:i})}function Lr(n){const e={},t=Bt(e);T.info("sendGppAcceptAllMessage: {}",e),Gn({messageType:"process_gpp_accept_all",data:t})}function Or(n){const e={},t=Bt(e);T.info("sendGppRejectAllMessage: {}",e),Gn({messageType:"process_gpp_reject_all",data:t})}function Dr(n){const{gppCountryState:e,activeGppPurposes:t,activeGppSubsections:i,categories:o,activeCategories:a}=n,r=ve("gpp.gppShowCategories"),l=ur(e.purposes||[],t,o,a,r);Gn({messageType:"submit_preferences",data:Gt(l)}),Er({activeGppPurposes:t,activeGppSubsections:i},e)}function Fr(n){const{gppCountryState:e,categories:t}=n,i=ve("gpp.gppShowCategories"),o=gr(e.purposes||[],t,i);Gn({messageType:"submit_preferences",data:Gt(o)}),Lr()}function Nr(n){Or()}function Ir(n,e){if(!e)return n;try{return new URL(n,e).href}catch(t){return T.warn("Invalid URL. Relative URL: {}, base URL: {}",n,e,t),n}}function Rr(n){try{const e=new URL(n);return e.origin+e.pathname.substring(0,e.pathname.lastIndexOf("/")+1)}catch(e){return T.warn("Invalid full URL: {}",n,e),null}}function vt(n){const e=hn(),t=/^(https?:)?\/\//;if(!e)return n;const i=e.endsWith("/")?e:e+"/",o=r=>{const l=r.trim();if(t.test(l)||l.startsWith("data:")||l.startsWith("#"))return r;try{return l.startsWith("/")?new URL(l,e).toString():i+l}catch(c){return T.warn("Failed to construct URL. Relative URL: {}, base URL: {}",l,e,c),r}},a=/\b(src|href)\s*=\s*(['"])((?:(?!\2).)*)\2/gi;return n.replace(a,(r,l,c,d)=>{const p=o(d);return`${l}=${c}${p}${c}`})}function Mr(n){return n!=null&&n.trim()!==""}function Ut(n){return!Mr(n)}function V(n,e=!1){return n==null?e:n.toLowerCase()==="true"||n==="1"}function ue(n){const e=document.createElement("textarea");return e.setAttribute("translate","no"),e.innerHTML=n,e.value}function Br(n){var o;const t=new DOMParser().parseFromString(`<div>${n}</div>`,"text/html");return t.querySelectorAll("img:not([alt])").forEach(a=>{a.setAttribute("alt","")}),((o=t.body.firstElementChild)==null?void 0:o.innerHTML)||n}function He(n){const e=ue(n),t=vt(e);return Br(t)}function Gt(n,e=!1){return n.length===0?e?"":"0":[...n].sort((t,i)=>t-i).join(",")}function Hr(n){return n.filter(e=>Ut(e.iabPurposeId))}function Ce(n){const{allCategories:e,dispatch:t}=n;let{level:i}=n;i<0&&(T.warn("[toggleActiveCategoriesToLargestIndexInclusive]: largestCategoryIndex < 0: {}",i),i=0),t(fo({level:i,allCategories:e}));const{activeCategories:o,activeVendors:a}=po(uo(i),e);return T.info("[Preferences - Slider Panel] Set slider level(and according actives categories and vendors): {}",i),{activeCategories:o,activeVendors:a}}const po=(n,e)=>{const t={},i={},o=new Set(n);return e.forEach((a,r)=>{o.has(r)?(t[a.id]=!0,a.vendors.forEach(l=>{i[xn(a.id,l.id)]=!0})):(t[a.id]=!1,a.vendors.forEach(l=>{i[xn(a.id,l.id)]=!1}))}),{activeCategories:t,activeVendors:i}},uo=n=>((!Number.isFinite(n)||n<0)&&(T.warn("[getConsecutiveArrayFromZero]: maxVal {} is invalid, use 0 instead.",n),n=0),[...Array(Math.floor(n)+1).keys()]);function jt(n,e,t=!1){const i=[];return e.forEach((o,a)=>{n[o.id]&&i.push(a)}),!t&&!i.includes(0)&&i.unshift(0),i.sort((o,a)=>o-a),i}function xn(n,e){return`${n}_${e}`}function zr(n,e){const t=jt(n,e);return t.length>0?Math.max(...t):0}function ot(n,e,t){let i=null;const o=zr(e,t);return n.level!==o&&(i={...n,level:o},T.info("[Preferences - Advanced Panel] Set slider level to: {} by active categories: {}",i.level,e)),i}function Wt(n){return n.map((e,t)=>({category:e,originalIndex:t})).filter(({category:e,originalIndex:t})=>e.vendors&&e.vendors.length>0||t===0)}function go(n,e){if(!n.categories||n.categories.length<=0)return T.info("[Disclosure] Category is null. Domain: {}",e.cmId),0;let t=n.categories,i;if(e.iab){i=t.length-1;for(let o=t.length-1;o>=0;o--){const a=t[o].iabPurposeId;if(Ut(a)){i=o;break}}}else return t=n.categories,t.length-1;return i}const Ct=async n=>{const{currentView:e,cmConfig:t,rootProps:i,dispatch:o,consent:a}=n;let r;e==="euDisclosurePanel"?r=yn.AGREE_AND_PROCEED:r=yn.ACCEPT_ALL;const l=Ce({level:go(t,i),allCategories:t.categories,dispatch:o});if(await(a==null?void 0:a.doConsent(r,l)),fn(i)){const c=t.config.gppCountryStates&&t.config.gppCountryStates.length>0?t.config.gppCountryStates[0]:void 0;c&&Fr({gppCountryState:c,categories:t.categories})}de("truste-click-accept-all-button",{})},$r=n=>{const{categoryId:e,categoryIndex:t,targetState:i,dispatch:o,allCategories:a}=n;de(`truste-click-toggle-category-${t}-${i?"on":"off"}`,{}),o(Yr({categoryId:e,categoryToggledOn:i,allCategories:a}))},Vr=n=>ue(n.displayText||n.name||"");function Vn(n,e){return n==="PRO"&&(e==null?void 0:e.custom_overlay_dpdp_required_switch)==="true"}const Ur="1",Gr={expandedCategories:{},activeCategories:{},activeVendors:{},expandedVendors:{},showAllDomainsVendors:{},activeTooltipVendor:null,sliderState:{heightList:[],levelHeightCheckList:[],level:0,sliderPanelHeight:0,sliderWrapperHeight:0,sliderTop:0},savedCategoriesConsent:{},savedVendorsConsent:{},activeStacks:{},activePurposes:{},activePurposeLegitimateInterests:{},activeSpecialFeatures:{},activeVendorConsents:{},activeVendorLegitimateInterests:{},activeGooglePartnerConsents:{},selectedIabTab:"categories_0",isProgrammaticScroll:!1,expandedPurposes:{},expandedSpecialPurposes:{},expandedFeatures:{},expandedSpecialFeatures:{},expandedPartners:{},expandedGooglePartners:{},selectedCategoryForDetails:{},showUnselectedWarning:!1,submitting:!1,activeGppPurposes:{},activeGppSubsections:{},expandedGppPurposes:{}},jr={toggleCategoryExpand:(n,e)=>({...n,expandedCategories:{...n.expandedCategories,[e]:!n.expandedCategories[e]}}),toggleCategory:(n,e)=>{const{categoryId:t,categoryToggledOn:i,allCategories:o}=e,a={...n.activeCategories,[t]:i},r={...n.activeVendors},l=o.find(f=>f.id===t);if(l!=null&&l.vendors)for(const f of l.vendors){const g=xn(t,f.id);r[g]=i}const c=ot(n.sliderState,a,o),u=Wt(o).some(({category:f})=>a[f.id]==null)?n.showUnselectedWarning:!1;return{...n,activeCategories:a,activeVendors:r,showUnselectedWarning:u,...c&&{sliderState:c}}},toggleVendor:(n,e)=>{const{categoryId:t,vendorId:i,vendorToggledOn:o,allCategories:a}=e,r=xn(t,i),l={...n.activeVendors,[r]:o},c=a.find(f=>f.id===t);let d=!1;c!=null&&c.vendors&&(d=c.vendors.some(f=>{const g=xn(t,f.id);return l[g]}));const p={...n.activeCategories,[t]:d},u=ot(n.sliderState,p,a);return{...n,activeVendors:l,activeCategories:p,...u&&{sliderState:u}}},toggleVendorExpand:(n,e)=>{const{categoryId:t,vendorId:i}=e,o=xn(t,i);return{...n,expandedVendors:{...n.expandedVendors,[o]:!n.expandedVendors[o]}}},toggleShowAllDomains:(n,e)=>{const{categoryId:t,vendorId:i}=e,o=xn(t,i);return{...n,showAllDomainsVendors:{...n.showAllDomainsVendors,[o]:!n.showAllDomainsVendors[o]}}},setActiveTooltipVendor:(n,e)=>{const t=e?xn(e.categoryId,e.vendorId):null;return{...n,activeTooltipVendor:t}},setActiveCategoriesAndVendors:(n,e)=>{const{activeCategories:t,activeVendors:i,allCategories:o}=e,a=ot(n.sliderState,t,o);return{...n,activeCategories:t,activeVendors:i,...a&&{sliderState:a}}},setSliderLevel:(n,e)=>{const{level:t}=e;if(n.sliderState.level===t)return n;const{activeCategories:i,activeVendors:o}=po(uo(t),e.allCategories);return{...n,sliderState:{...n.sliderState,level:t},activeCategories:i,activeVendors:o}},setSliderUiState:(n,e)=>{const t=(o,a)=>{switch(o){case"heightList":case"levelHeightCheckList":return Array.isArray(a)&&a.every(r=>typeof r=="number"&&!isNaN(r));case"sliderTop":case"sliderPanelHeight":case"sliderWrapperHeight":return typeof a=="number"&&!isNaN(a);default:return!1}},i={...n.sliderState};for(const[o,a]of Object.entries(e))if(t(o,a)){if(n.sliderState[o]===a)continue;i[o]=a}return{...n,sliderState:i}},syncSavedAndActiveConsent:(n,e)=>{const{activeCategories:t,activeVendors:i}=e;return{...n,savedCategoriesConsent:structuredClone(t),savedVendorsConsent:structuredClone(i)}},revertActiveStateToSaved:(n,e)=>{const{savedCategoriesConsent:t,savedVendorsConsent:i}=e;return{...n,activeCategories:structuredClone(t),activeVendors:structuredClone(i)}},setSelectedCategoryForDetails:(n,e)=>({...n,selectedCategoryForDetails:e}),setShowUnselectedWarning:(n,e)=>({...n,showUnselectedWarning:e}),setSubmitting:(n,e)=>({...n,submitting:e}),toggleStack:(n,e)=>{const{stackId:t,stackToggledOn:i,iabData:o}=e,a={...n.activeStacks,[t]:i};if(!o)return{...n,activeStacks:a};const r=o.stacksById.get(Number(t));if(!r)return{...n,activeStacks:a};const l={...n.activePurposes},c={...n.activePurposeLegitimateInterests},d={...n.activeSpecialFeatures},p={...n.activeVendorConsents},u={...n.activeVendorLegitimateInterests};for(const f of r.purposes)l[String(f)]=i,o.legIntPurposes.has(f)&&(c[String(f)]=i);for(const f of r.specialFeatures)d[String(f)]=i;for(const f of o.vendorsById.values()){const g=String(f.id);i?(f.purposes.some(x=>r.purposes.includes(x))&&(p[g]=!0),f.legIntPurposes.some(x=>r.purposes.includes(x))&&(u[g]=!0)):(f.purposes.some(x=>l[String(x)]===!0)||(p[g]=!1),f.legIntPurposes.some(x=>c[String(x)]===!0)||(u[g]=!1))}return{...n,activeStacks:a,activePurposes:l,activePurposeLegitimateInterests:c,activeSpecialFeatures:d,activeVendorConsents:p,activeVendorLegitimateInterests:u}},togglePurpose:(n,e)=>{const{purposeId:t,purposeToggledOn:i,iabData:o}=e,a={...n.activePurposes,[t]:i};if(!o)return{...n,activePurposes:a};const r={...n.activePurposeLegitimateInterests},l={...n.activeVendorConsents},c={...n.activeVendorLegitimateInterests},d=Number(t);o.legIntPurposes.has(d)&&(r[t]=i);for(const p of o.vendorsById.values()){const u=String(p.id);i?(p.purposes.includes(d)&&(l[u]=!0),p.legIntPurposes.includes(d)&&(c[u]=!0)):(p.purposes.includes(d)&&!p.purposes.some(f=>a[String(f)]===!0)&&(l[u]=!1),p.legIntPurposes.includes(d)&&!p.legIntPurposes.some(f=>a[String(f)]===!0||r[String(f)]===!0)&&(c[u]=!1))}return{...n,activePurposes:a,activePurposeLegitimateInterests:r,activeVendorConsents:l,activeVendorLegitimateInterests:c}},toggleLegitimateInterestPurpose:(n,e)=>{const{purposeId:t,purposeToggledOn:i}=e;return{...n,activePurposeLegitimateInterests:{...n.activePurposeLegitimateInterests,[t]:i}}},toggleSpecialFeature:(n,e)=>{const{specialFeatureId:t,specialFeatureToggledOn:i}=e;return{...n,activeSpecialFeatures:{...n.activeSpecialFeatures,[t]:i}}},toggleVendorConsent:(n,e)=>{const{vendorId:t,consentToggledOn:i}=e;return{...n,activeVendorConsents:{...n.activeVendorConsents,[t]:i}}},toggleVendorLegitimateInterest:(n,e)=>{const{vendorId:t,legitimateInterestToggledOn:i}=e;return{...n,activeVendorLegitimateInterests:{...n.activeVendorLegitimateInterests,[t]:i}}},toggleGooglePartnerConsent:(n,e)=>{const{partnerId:t,consentToggledOn:i}=e;return{...n,activeGooglePartnerConsents:{...n.activeGooglePartnerConsents,[t]:i}}},initializeIabConsents:(n,e)=>{const{activeStacks:t,activePurposes:i,activePurposeLegitimateInterests:o,activeSpecialFeatures:a,activeVendorConsents:r,activeVendorLegitimateInterests:l,activeGooglePartnerConsents:c}=e;return{...n,activeStacks:t,activePurposes:i,activePurposeLegitimateInterests:o,activeSpecialFeatures:a,activeVendorConsents:r,activeVendorLegitimateInterests:l,activeGooglePartnerConsents:c}},setSelectedIabTab:(n,{tabId:e,isProgrammaticScroll:t=!0})=>({...n,selectedIabTab:e,isProgrammaticScroll:t}),togglePurposeExpand:(n,e)=>({...n,expandedPurposes:{...n.expandedPurposes,[e]:!n.expandedPurposes[e]}}),expandAllPurposes:(n,e)=>{const t={...n.expandedPurposes};return e.forEach(i=>{t[i]=!0}),{...n,expandedPurposes:t}},collapseAllPurposes:(n,e)=>{const t={...n.expandedPurposes};return e.forEach(i=>{t[i]=!1}),{...n,expandedPurposes:t}},toggleSpecialPurposeExpand:(n,e)=>({...n,expandedSpecialPurposes:{...n.expandedSpecialPurposes,[e]:!n.expandedSpecialPurposes[e]}}),expandAllSpecialPurposes:(n,e)=>{const t={...n.expandedSpecialPurposes};return e.forEach(i=>{t[i]=!0}),{...n,expandedSpecialPurposes:t}},collapseAllSpecialPurposes:(n,e)=>{const t={...n.expandedSpecialPurposes};return e.forEach(i=>{t[i]=!1}),{...n,expandedSpecialPurposes:t}},toggleFeatureExpand:(n,e)=>({...n,expandedFeatures:{...n.expandedFeatures,[e]:!n.expandedFeatures[e]}}),expandAllFeatures:(n,e)=>{const t={...n.expandedFeatures};return e.forEach(i=>{t[i]=!0}),{...n,expandedFeatures:t}},collapseAllFeatures:(n,e)=>{const t={...n.expandedFeatures};return e.forEach(i=>{t[i]=!1}),{...n,expandedFeatures:t}},toggleSpecialFeatureExpand:(n,e)=>({...n,expandedSpecialFeatures:{...n.expandedSpecialFeatures,[e]:!n.expandedSpecialFeatures[e]}}),expandAllSpecialFeatures:(n,e)=>{const t={...n.expandedSpecialFeatures};return e.forEach(i=>{t[i]=!0}),{...n,expandedSpecialFeatures:t}},collapseAllSpecialFeatures:(n,e)=>{const t={...n.expandedSpecialFeatures};return e.forEach(i=>{t[i]=!1}),{...n,expandedSpecialFeatures:t}},togglePartnerExpand:(n,e)=>({...n,expandedPartners:{...n.expandedPartners,[e]:!n.expandedPartners[e]}}),expandAllPartners:(n,e)=>{const t={...n.expandedPartners};return e.forEach(i=>{t[i]=!0}),{...n,expandedPartners:t}},collapseAllPartners:(n,e)=>{const t={...n.expandedPartners};return e.forEach(i=>{t[i]=!1}),{...n,expandedPartners:t}},toggleGooglePartnerExpand:(n,e)=>({...n,expandedGooglePartners:{...n.expandedGooglePartners,[e]:!n.expandedGooglePartners[e]}}),expandAllGooglePartners:(n,e)=>{const t={...n.expandedGooglePartners};return e.forEach(i=>{t[i]=!0}),{...n,expandedGooglePartners:t}},collapseAllGooglePartners:(n,e)=>{const t={...n.expandedGooglePartners};return e.forEach(i=>{t[i]=!1}),{...n,expandedGooglePartners:t}},toggleGppPurpose:(n,e)=>{const t={...n.activeGppPurposes},i={...n.activeGppSubsections};t[e.purposeId]=e.enabled;const o=Object.keys(n.activeGppSubsections).filter(a=>a.startsWith(`${e.purposeId}_`));return o.length>0&&o.forEach(a=>{i[a]=e.enabled}),{...n,activeGppPurposes:t,activeGppSubsections:i}},toggleGppSubsection:(n,e)=>{const t=`${e.purposeId}_${e.subsectionName}`,i={...n.activeGppSubsections},o={...n.activeGppPurposes};i[t]=e.enabled;const a=Object.keys(i).filter(r=>r.startsWith(`${e.purposeId}_`));if(a.length>0){const r=a.map(d=>i[d]),l=r.some(d=>d===Wn.SHOW_NO);r.every(d=>d===Wn.SHOW_YES)?o[e.purposeId]=Wn.SHOW_YES:l?o[e.purposeId]=Wn.SHOW_NO:o[e.purposeId]=Wn.UNSET}return{...n,activeGppPurposes:o,activeGppSubsections:i}},toggleGppPurposeExpand:(n,e)=>({...n,expandedGppPurposes:{...n.expandedGppPurposes,[e]:!n.expandedGppPurposes[e]}}),initializeGppConsents:(n,e)=>{const t={...n.activeGppPurposes},i={...n.activeGppSubsections},o={...n.expandedGppPurposes};return e.filter(a=>a.value!==Re.NOT_APPLICABLE).forEach(a=>{a.name in t||(t[a.name]=!1),a.id in o||(o[a.id]=!1),a.subsections&&a.subsections.length>0&&a.subsections.filter(r=>r.enabled).forEach(r=>{const l=`${a.name}_${r.name}`;l in i||(i[l]=!1)})}),{...n,activeGppPurposes:t,activeGppSubsections:i,expandedGppPurposes:o}},initializeGppConsentsFromCmpApi:(n,e)=>{const{cmpApiData:t,stateCode:i}=e,o={...n.activeGppPurposes},a={...n.activeGppSubsections},r=Object.keys(o).length===0,l=Object.keys(a).length===0;T.info("initializeGppConsentsFromCmpApi - cmpApiData: {}, stateCode: {}",JSON.stringify(t),i);const c=yt(i),d=c.sensitiveData,p=c.knownChild;return t&&typeof t=="object"&&Object.entries(t).forEach(([u,f])=>{if(!(u==="Version"||u==="Gpc")){if(typeof f=="number"){const g=xi(f);(r||u in o)&&(o[u]=g)}else if(Array.isArray(f)){let g=[];u==="SensitiveDataProcessing"?g=d:u==="KnownChildSensitiveDataConsents"&&(g=p);let x=!1;if(f.forEach((b,h)=>{const m=g[h];if(m){const _=`${u}_${m}`,k=xi(b);(l||_ in a)&&(a[_]=k),b===er.OPT_OUT&&(x=!0)}}),r||u in o){const b=x?Wn.SHOW_NO:Wn.SHOW_YES;o[u]=b}}}}),T.info("GPP initialization complete - {} purposes, {} subsections",Object.keys(o).length,Object.keys(a).length),{...n,activeGppPurposes:o,activeGppSubsections:a}}},Yt=_e({name:"preferences",initialState:Gr,reducers:jr}),{toggleCategoryExpand:Wr,toggleCategory:Yr,toggleVendor:qr,toggleVendorExpand:Kr,toggleShowAllDomains:Qr,setActiveTooltipVendor:Zr,setSliderUiState:Xr,setSliderLevel:fo,setActiveCategoriesAndVendors:bo,syncSavedAndActiveConsent:ho,revertActiveStateToSaved:Jr,setSelectedCategoryForDetails:Uc,setShowUnselectedWarning:Gc,setSubmitting:jc,toggleStack:Wc,togglePurpose:Yc,toggleLegitimateInterestPurpose:qc,toggleSpecialFeature:Kc,toggleVendorConsent:Qc,toggleVendorLegitimateInterest:Zc,initializeIabConsents:Xc,setSelectedIabTab:Jc,togglePurposeExpand:np,expandAllPurposes:ep,collapseAllPurposes:tp,toggleSpecialPurposeExpand:ip,expandAllSpecialPurposes:op,collapseAllSpecialPurposes:ap,toggleFeatureExpand:rp,expandAllFeatures:sp,collapseAllFeatures:lp,toggleSpecialFeatureExpand:cp,expandAllSpecialFeatures:pp,collapseAllSpecialFeatures:dp,togglePartnerExpand:up,expandAllPartners:gp,collapseAllPartners:fp,toggleGooglePartnerExpand:bp,expandAllGooglePartners:hp,collapseAllGooglePartners:mp,toggleGooglePartnerConsent:xp,toggleGppPurpose:ns,toggleGppSubsection:es,toggleGppPurposeExpand:ts,initializeGppConsents:is,initializeGppConsentsFromCmpApi:_p}=Yt.actions,os={configSlice:Jn.initialState,globalStateSlice:Dt.initialState,activityLoggingSlice:ye.initialState,preferencesSlice:Yt.initialState,uiStateSlice:Ft.initialState};function as(n,e){return{configSlice:Jn.reducer(n.configSlice,e),globalStateSlice:Dt.reducer(n.globalStateSlice,e),activityLoggingSlice:ye.reducer(n.activityLoggingSlice,e),preferencesSlice:Yt.reducer(n.preferencesSlice,e),uiStateSlice:Ft.reducer(n.uiStateSlice,e)}}const Ue=$i(void 0),rs=({children:n})=>{const[e,t]=$e(as,os),i=R(null);R(e),W(()=>{oe.setAppState(e)},[e]),W(()=>{},[]);const o=$(()=>r=>{i.current&&ls(),t(r)},[]),a=$(()=>({state:e,dispatch:o}),[e,o]);return s(Ue.Provider,{value:a,children:n})};function y(n){const e=xe(Ue);if(!e)throw new Error("useSelector must be used within StoreProvider");return $(()=>n(e.state),[n,e.state])}function rn(){const n=xe(Ue);if(!n)throw new Error("useDispatch must be used within StoreProvider");return n.dispatch}const ss=()=>{const n=xe(Ue);if(!n)throw new Error("useStore must be used within StoreProvider");return n},ls=()=>typeof window<"u"&&window.__REDUX_DEVTOOLS_EXTENSION__&&Ga();function cs(n,e,t){let i=n;if(e&&(i+=e),t&&Object.keys(t).length>0){const o=new URLSearchParams(t).toString();i+=`?${o}`}return i}async function ps(n){const{baseUrl:e,endpoint:t,method:i,queryParams:o,requestBody:a,onStart:r,onError:l,onSuccess:c}=n,d=cs(e,t,o),p={};(i==="POST"||i==="PUT")&&(p["Content-Type"]="application/json");try{r==null||r(i,d,t);const u=await fetch(d,{method:i,headers:p,body:a?JSON.stringify(a):void 0,keepalive:!0});if(!u.ok){const f=`HTTP Error ${u.status}: ${u.statusText}`;T.error(f),l==null||l(f,u.status);return}c(u)}catch(u){l==null||l(u)}}const ds={callRestApi:ps};function Ge(){const n=y(t=>t.globalStateSlice.advRootProps),e=y(t=>t.globalStateSlice.proRootProps);return n.cmType==="ADV"?n:e}const mo=3e3,us=n=>()=>{n(ye.actions.logSendStarted())},gs=n=>()=>{n(ye.actions.logSendSuccess())},fs=n=>e=>{n(ye.actions.logSendFailed(Ht(e)))};function bs(n,e){return n!==yn.SUBMIT_PREFERENCES&&e!==null&&e.length>0?(T.error("[ActivityLog] actionType: {} is not OPT_IN, but categories is not null or empty. Changing to OPT_IN.",n,e),n=yn.SUBMIT_PREFERENCES):n===yn.SUBMIT_PREFERENCES&&(e===null||e.length===0)&&T.warn("[ActivityLog] actionType is OPT_IN but categories are null or empty."),n}async function ge(n,e,t){const i=bs(n,e),o=hs(e,t);T.info("[ActivityLog] Activity: actionType: {}, computeCategoriesInt: {}",i,o),await Pr(i,o)}function hs(n,e){if(!n||n.length===0)return;const t=new Set(n.map(a=>a.name)),i=je(e);let o=0;return i.forEach((a,r)=>{t.has(a.name)&&(o+=1<<r)}),o===0?void 0:o}async function ms(n,e){const t=xs(n),{actionType:i}=t;i!==yn.SUBMIT_PREFERENCES&&await ge(i,null,e),await _s(t,e);const o=Cs(t,e);await ge(yn.SUBMIT_PREFERENCES,o,e),await ws(o,e)}function xs(n){const e=[...new Set(n.optIns)].sort((t,i)=>t-i);return{...n,optIns:e}}async function _s(n,e){const{actionType:t,optIns:i}=n;if(t!==yn.SUBMIT_PREFERENCES)return;const o=je(e);(e.rootProps.cmType==="ADV"?o:Wt(o)).length===i.length?await ge(wt.OPT_IN_ALL,null,e):i.length===1&&i[0]===0&&await ge(wt.OPT_OUT_ALL,null,e)}const ys=function(n){return n.reduce((e,t)=>e+(1<<t),0)};async function ws(n,e){var p,u,f,g,x,b,h;if(oe.isPreview()){T.info("[ActivityLog] Skipping logConsent in preview mode.");return}const{rootProps:t,dispatch:i}=e,o=Vt(),a=je(e),r=new Set(n.map(m=>m.name)),l=a.map((m,_)=>r.has(m.name)?_:-1).filter(m=>m!==-1),c={country:t.country,state:t.state,behavior:o,uid:t.uid,categoryOrgIdsArr:n.flatMap(m=>m.vendors.map(_=>_.intId)),domain:t.cmId,session:t.session,userType:t.userType,categories:ys(l)};let d=!1;if(typeof window<"u"&&((p=window.truste)!=null&&p.cd))try{const m=window.truste,_=(f=(u=m.eu)==null?void 0:u.user)==null?void 0:f.authMac,k=(x=(g=m.eu)==null?void 0:g.user)==null?void 0:x.id,A=(h=(b=m.eu)==null?void 0:b.user)==null?void 0:h.authSecretId;k&&A&&_&&(c.userId=k,c.authSecretId=A,c.authMac=_,d=!0)}catch(m){T.warn("[ActivityLog] Failed to generate cross domain parameters:",m)}await ds.callRestApi({baseUrl:hn(),endpoint:ro.ENDPOINT_SEND_LOG_CONSENT,method:"POST",requestBody:c,onStart:us(i),onSuccess:()=>{gs(i)(),d&&de("trustarc-consent-save-success",{})},onError:(m,_)=>{fs(i)(m),d&&de("trustarc-consent-save-failed",_!=null?{status:_}:{error:"network"})}})}function vs(n,e){return n.filter((t,i)=>e.includes(i))}function Cs(n,e){const t=je(e);return vs(t,n.optIns)}function je(n){const{rootProps:e,cmConfig:t}=n;return e.cmType==="ADV"&&e.iab?t.categoriesWithIab:t.categories}function qt(){const n=Ge(),e=y(o=>o.configSlice.config),t=rn(),i={rootProps:n,cmConfig:e,dispatch:t};return{logInformationalActivity:o=>ge(o,null,i),logConsentActivity:o=>ms(o,i)}}function Ss(n,e){const t=[...n.children].filter(i=>i.tagName===e.tagName);if(t.length>0){const i=t[t.length-1];n.insertBefore(e,i.nextSibling)}else n.insertBefore(e,n.firstChild)}function ks(n){const e=xo();e&&n.setAttribute("nonce",e)}function xo(){const n=document.querySelector("script[nonce], style[nonce]");return(n==null?void 0:n.nonce)||(n==null?void 0:n.getAttribute("nonce"))}function _o(n,e){const{tag:t,dataAttribute:i,insertWithPrecedence:o}=e,a=`${t}[${i}="true"]`;let r=n.querySelector(a);return r||(T.info("Creating new {} element with attribute {} in shadow DOM",t,i),r=document.createElement(t),r.setAttribute(i,"true"),ks(r),o?Ss(n,r):n.appendChild(r)),r}const As=/^https?:\/\//i;function Ts(n,e){if(!e)return;const t=_o(n,{tag:"script",dataAttribute:"data-on-load-script"}),i=e.trim();As.test(i)?(t.onerror=()=>T.error("[CCM] Failed to load external on-load script from {}",i),t.src=i):(t.text=e,t.type="text/javascript")}function Ps(n){if(typeof n.checkVisibility=="function")return n.checkVisibility({checkVisibilityCSS:!0,checkOpacity:!0});if(window.getComputedStyle(n).visibility==="hidden")return!1;for(let e=n;e;e=e.parentElement){const t=window.getComputedStyle(e);if(t.display==="none"||t.opacity==="0")return!1}return!0}function Es(n){return"imported-style-"+n.replace(/^https?:\/\//,"").replace(/[^\w\d]/g,"-").replace(/-+/g,"-").replace(/^-|-$/g,"").toLowerCase()}function yo(n){if(!n.current)return null;const e=n.current.getRootNode();return e instanceof ShadowRoot?e:(T.warn("Component is not inside a shadow root"),null)}function Ls(n){return/(privacyPolicyAnchor|cookiePolicyAnchor|trackerInformationAnchor|data-(cookie|privacy)-policy-link)/.test(n)}function wo(n,e,t){if(!n)return"";if(!Ls(n))return vt(n);const o=new DOMParser().parseFromString(n,"text/html");return yi(o,"a#privacyPolicyAnchor, a[data-privacy-policy-link]",{href:e.privacypolicylink??(t==null?void 0:t.privacyPolicy)??"",text:e.privacypolicylinktext}),yi(o,"a#cookiePolicyAnchor, a[data-cookie-policy-link]",{href:e.cookieLink,text:e.cookieLinkText}),Os(o,e.locale),vt(o.head.innerHTML+o.body.innerHTML)}function yi(n,e,t){n.querySelectorAll(e).forEach(o=>{t.href&&(o.href=t.href),t.text&&(o.textContent=t.text)})}function Os(n,e){const t=n.querySelector("a#trackerInformationAnchor");if(!t)return;const i=t.getAttribute("href");if(!i||i.includes("locale="))return;const o=i.includes("?")?"&":"?";t.setAttribute("href",i+o+"locale="+e)}function Kt(n){let e=n.trim();return e?(e=e.replace(/\s+/g," "),e=e.replace(/\s+;/g,";"),e.endsWith(";")||(e+=";"),e=e.replace(/;+/g,";"),e):""}function Ds(n){const e=Kt(n);if(!e)return"";let t=5381;for(let o=0;o<e.length;o++){const a=e.charCodeAt(o);t=(t<<5)+t+a,t=t|0}return`inline-${Math.abs(t).toString(36)}`}function Fs(n){const e=[];return n.forEach((t,i)=>{const o=Kt(t);if(o){const a=o.split(";").map(r=>r.trim()).filter(Boolean).map(r=>r.includes("!important")?r:r+" !important").join("; ");e.push(`.${i} { ${a}; }`)}}),e.join(`
`)}function Ns(n){const e=new Map;if(!n||!n.trim())return{html:n||"",css:"",classMap:e};const t=/\s+style\s*=\s*(['"])((?:(?!\1).)*)\1/gi;let i=n;i=i.replace(t,(r,l,c)=>{const d=Kt(c);if(!d)return"";const p=Ds(d);return p?(e.has(p)||e.set(p,d),` __STYLE_EXTRACTED_${p}__`):(console.warn("[inline-style-extractor] Failed to generate hash for style:",c),r)});const o=/__STYLE_EXTRACTED_([a-z0-9-]+)__/gi;i=i.replace(/<([a-z][a-z0-9]*)\s+([^>]*?)>/gi,(r,l,c)=>{if(!c.match(o))return r;const p=[];let u=c.replace(o,(x,b)=>(p.push(b),""));u=u.replace(/\s+/g," ").trim();const f=/\bclass\s*=\s*(['"])([^'"]*)\1/i,g=u.match(f);if(g){const x=g[2],b=x?`${x} ${p.join(" ")}`:p.join(" ");u=u.replace(f,`class="${b}"`)}else{const x=`class="${p.join(" ")}"`;u=u?`${x} ${u}`:x}return`<${l} ${u}>`}),i=i.replace(o,"");const a=Fs(e);return{html:i,css:a,classMap:e}}function Is(n){const e=[];if(!n||!n.trim())return{html:n||"",css:"",classMap:new Map};const t=/<style\b[^>]*>([\s\S]*?)<\/style>/gi;let i=n;return i=i.replace(t,(o,a)=>{const r=a.trim();return r&&e.push(r),""}),i=i.replace(/\n\s*\n/g,`
`).trim(),{html:i,css:e.join(`
`),classMap:new Map}}function Rs(n){const e=Is(n),t=Ns(e.html),i=[e.css,t.css].filter(o=>o.trim()).join(`
`);return{html:t.html,css:i}}const Ms=(n,e=!1)=>{const t=document.createElement("div");return t.setAttribute("translate","no"),t.innerHTML=n,e||["script","iframe","object","embed","form"].forEach(o=>{var r;const a=t.getElementsByTagName(o);for(let l=a.length-1;l>=0;l--){const c=a[l];(r=c.parentNode)==null||r.removeChild(c)}}),t.innerHTML},Bs=({containerRef:n,html:e,css:t,eventHandlers:i={},allowScripts:o=!1,onRendered:a})=>{const[r,l]=_n(!1),c=$(()=>Rs(e),[e]),d=ln(()=>Ms(c.html,o),[c.html,o]),p=$(()=>[c.css,t].filter(Boolean).join(`

`),[c.css,t]),u=ln(()=>{if(!p)return"";const g=xo();return`<style ${g?`nonce="${g}"`:""}>${p}</style>`},[p]),f=ln(g=>(Object.entries(i).forEach(([x,b])=>{g.querySelectorAll(x).forEach(m=>{Object.entries(b).forEach(([_,k])=>{m.addEventListener(_,k)})})}),()=>{Object.entries(i).forEach(([x,b])=>{g.querySelectorAll(x).forEach(m=>{Object.entries(b).forEach(([_,k])=>{m.removeEventListener(_,k)})})})}),[i]);return W(()=>{if(!n.current)return;n.current.setAttribute("translate","no"),n.current.innerHTML=d()+u();const g=f(n.current);return l(!0),g},[d,f]),W(()=>{r&&a&&a()},[r,a]),null},U=Xi(Bs,(n,e)=>n.html===e.html&&n.css===e.css&&n.allowScripts===e.allowScripts&&n.eventHandlers===e.eventHandlers&&n.onRendered===e.onRendered),We=()=>({createEventHandlers:ln(e=>{const t={};return Object.entries(e).forEach(([i,o])=>{t[i]={},Object.entries(o).forEach(([a,r])=>{t[i][a]=l=>{const c=l.target;r(l,c)}})}),t},[])}),Hs=(n,e)=>n.action==="show2ndoptinstep"?"twoStepPanel":n.action==="show_loyalty_message"?"loyaltyMessage":n.action==="required"?"sliderPanel":Bn(n)?"advancedPanel":fn(n)?"gppPanel":n.action==="notice"?"euDisclosurePanel":n.action==="prefmgr"?ze(n,e):n.action==="advanced"?"advancedPanel":co()&&!ve("isReconsentEvent")?ze(n,e):"euDisclosurePanel";function ze(n,e){return V(e.preftable_show_advance_panel_value_key)||Bn(n)?"advancedPanel":"sliderPanel"}const zs="Submit Preferences",vo="Accept All",Co="Decline All",$s="Cancel",wi="Close",Vs="View Cookie Settings &raquo;",Us="Ask me later",Gs="Back",So="No",ko="Yes",js="Basic Settings",Ws="Advanced Settings",Ao='<span class="disableSubmitMessage">Please complete your choices before submitting.</span>',Ys="cancel,submit,accept_all,decline_all,advanced_pref",qs="accept_all,decline_all,ask_me_later,show_pref_mgr",Ks="submit,accept_all,decline_all",Qs="cancel",at="Show All",vi="Hide",rt="Privacy Policy",Zs="Company",Xs="Domain",Js="Domain(s):",nl="Opt-out",To="No cookies exist for this category",el="expressed",tl="Reject the {CATEGORY_NAME_DO_NOT_REMOVE} from",il="Accept the {CATEGORY_NAME_DO_NOT_REMOVE} from",ol="Collapse table row",al="Expand table row",rl="view_cookie,category_switch,pref_panel",sl=380,ll="Accept All Cookies",Po="IN",Eo="OUT",Ci="en",cl="Terms and Conditions",pl="Under Review",dl="Active";`${hn()}`;const ul="Example Purposes";async function Lo(n,e){let t;const i=new Promise(o=>{t=setTimeout(o,e)});try{await Promise.race([n,i])}catch{}finally{clearTimeout(t)}}let Oe=null;function gl(n){return new Promise(e=>{Oe=e,n(un("loyaltyMessage"))})}function fl(n){Oe&&(Oe(n),Oe=null)}function bl(n,e,t){return{cpraConsent:n,isNoticeFlow:e,cpraSource:t}}const hl=(n,e)=>{const t=Object.keys(n);if(t.length<e.length)return!0;for(const i of t)if(n[i]===!1)return!0;return!1};function ml(n,e,t){return!!t.is_cpra&&!!t.is_finprog&&hl(n,e)}let De=null;function xl(n){const{cmType:e,dispatch:t}=n;return new Promise(i=>{De=i,t(un(e==="ADV"?"twoStepPanel":"proTwoStepPanel"))})}function st(n){De&&(De(n),De=null)}const Si=(n,e)=>{for(const t of Object.keys(e))if(e[t]===!0&&n[t]!==!0)return!0;return!1};function _l(n){const{twoStepEnabled:e,oldActiveCategories:t,newActiveCategories:i,oldActiveVendors:o,newActiveVendors:a}=n;if(!e||Object.keys(t).length===0)return!1;const r=Si(t,i),l=Si(o,a);return r||l}function Oo(){const n=rn(),e=y(c=>c.configSlice.config),t=Ge(),i=y(c=>c.preferencesSlice.savedCategoriesConsent),o=y(c=>c.preferencesSlice.savedVendorsConsent),a=y(c=>c.globalStateSlice.currentView),r=qt();return{doConsent:async(c,d)=>{let{activeCategories:p,activeVendors:u}=d;if(t.cmType==="ADV"&&ml(p,e.categories,t)){const{activeCategoriesUpdated:h,activeVendorsUpdated:m}=await gl(n);Object.keys(h).length>0&&(p=h,u=m)}else if(_l({twoStepEnabled:t.cmType==="ADV"?t.twoStep:e.config.secondStepOptIn,oldActiveCategories:i,newActiveCategories:p,oldActiveVendors:o,newActiveVendors:u})&&!await xl({cmType:t.cmType,dispatch:n}))return;xr({allCategories:e.categories,activeCategories:p,activeVendors:u,cmType:t.cmType,contents:e.contents},n);const f=Vn(t.cmType,e.contents),g=jt(p,e.categories,f);if(g.length>0){const h=r.logConsentActivity({actionType:c,optIns:g}).catch(m=>{T.error("Failed to log consent activity: {}",m)});await Lo(h,mo)}const x=a==="euDisclosurePanel"&&V(e.contents.disclosure_show_optoutdonemessage_key,!0),b=a!=="euDisclosurePanel"&&V(e.contents.preftable_show_optout_done_message_key,!0);t.cmType==="ADV"&&(x||b)?(Gn({messageType:Kn.CHANGE_PANEL,data:"0"}),n(un("optoutDoneMessage"))):En()}}}function Zn(){const n=y(t=>t.globalStateSlice.currentView),e=y(t=>t.configSlice.config.contents.preftable_granular_scroll_ui_key);return n==="advancedPanel"&&V(e)}function Do(n,e){return(n==="advancedPanel"||n==="sliderPanel")&&e?"advancePanelHeader":"cookieMainHeader"}function yl(n,e){if(!Nt(n))return Do(n,e)}function wl(n,e=50,t=500){return new Promise(i=>{const o=performance.now();let a=o;const r=new MutationObserver(()=>{a=performance.now()});r.observe(n,{childList:!0,subtree:!0,attributes:!0});const l=()=>{const c=performance.now();if(c-a>=e||c-o>t)return r.disconnect(),i();requestAnimationFrame(l)};requestAnimationFrame(l)})}async function Ye(n,e,t=!0){var r,l,c,d,p,u;if(!n||!j(e)&&!Tr())return;await wl(n);const i=n.querySelector("#trustarc-internal-close-button");if(!i)return;const o=Qn(hr),a=Qn("truste.eu.initCloseButtonLogic");if(!(!o||!a))try{const f=(l=(r=window.truste)==null?void 0:r.eu)==null?void 0:l.bindMap,g="cm";if(!((d=(c=window.truste)==null?void 0:c.eu)==null?void 0:d.shouldShowCloseButton(f,g,e.action))){i.style.setProperty("visibility","hidden","important"),i.tabIndex=-1;return}(u=(p=window.truste)==null?void 0:p.eu)==null||u.initCloseButtonLogic(i,mr(),g,e.action),i.classList.add("ta-modal-tertiary-btn"),i.style.cssText+="; visibility: visible !important; display: block !important",t&&requestAnimationFrame(()=>{i.focus(),i.classList.add("programmatic-focus");const b=()=>{i.classList.remove("programmatic-focus"),i.removeEventListener("blur",b)};i.addEventListener("blur",b)})}catch(f){console.warn("Failed to apply close button visibility:",f)}}function vl(){return s("svg",{"aria-hidden":"true",width:"14",height:"14",viewBox:"0 0 15 14",fill:"none",xmlns:"http://www.w3.org/2000/svg",children:[s("g",{clipPath:"url(#clip0_13990_35920)",children:[s("path",{d:"M13.9018 13.1911H1.09816C0.99329 13.1908 0.890283 13.1634 0.799143 13.1115C0.708003 13.0596 0.631826 12.9851 0.578011 12.895C0.526854 12.8033 0.5 12.7 0.5 12.595C0.5 12.4899 0.526854 12.3866 0.578011 12.2949L6.97985 1.09165C7.03679 1.00582 7.11408 0.935405 7.20484 0.886698C7.2956 0.83799 7.397 0.8125 7.5 0.8125C7.603 0.8125 7.7044 0.83799 7.79516 0.886698C7.88592 0.935405 7.96321 1.00582 8.02015 1.09165L14.422 12.2949C14.4731 12.3866 14.5 12.4899 14.5 12.595C14.5 12.7 14.4731 12.8033 14.422 12.895C14.3682 12.9851 14.292 13.0596 14.2009 13.1115C14.1097 13.1634 14.0067 13.1908 13.9018 13.1911ZM2.13046 11.9908H12.8695L7.5 2.59608L2.13046 11.9908Z",fill:"#816204"}),s("path",{d:"M7.49996 8.78984C7.34143 8.78777 7.18997 8.72387 7.07787 8.61176C6.96576 8.49965 6.90186 8.3482 6.89979 8.18967V5.38886C6.89979 5.22969 6.96302 5.07703 7.07557 4.96448C7.18813 4.85192 7.34078 4.78869 7.49996 4.78869C7.65914 4.78869 7.81179 4.85192 7.92435 4.96448C8.0369 5.07703 8.10013 5.22969 8.10013 5.38886V8.18967C8.09806 8.3482 8.03416 8.49965 7.92205 8.61176C7.80995 8.72387 7.65849 8.78777 7.49996 8.78984Z",fill:"#816204"}),s("path",{d:"M7.49996 11.1905C7.34143 11.1885 7.18997 11.1246 7.07787 11.0125C6.96576 10.9003 6.90186 10.7489 6.89979 10.5904V10.1902C6.89979 10.0311 6.96302 9.87841 7.07557 9.76586C7.18813 9.6533 7.34078 9.59007 7.49996 9.59007C7.65914 9.59007 7.81179 9.6533 7.92435 9.76586C8.0369 9.87841 8.10013 10.0311 8.10013 10.1902V10.5904C8.09806 10.7489 8.03416 10.9003 7.92205 11.0125C7.80995 11.1246 7.65849 11.1885 7.49996 11.1905Z",fill:"#816204"})]}),s("defs",{children:s("clipPath",{id:"clip0_13990_35920",children:s("rect",{width:"14",height:"14",fill:"white",transform:"translate(0.5)"})})})]})}const Cl=()=>{const n=R(null),e=y(o=>o.configSlice.config),t=y(o=>o.globalStateSlice.currentView);return t==="advancedPanel"||t==="iabManageSettingsPanel"?s("div",{className:"disabledSubmitMessageAccessible",id:"button-warning-messages",role:"alert",children:[s("div",{className:"warningIconContainer","aria-hidden":"true",children:s(vl,{})}),s("div",{className:"gwt-HTML",ref:n,children:s(U,{containerRef:n,html:e.contents.preftable_disabled_submit_message_key||Ao})})]}):null};function qn(n){return{onKeyDown:e=>{e.key==="Enter"&&(e.stopPropagation(),e.preventDefault(),n()),e.key===" "&&(e.stopPropagation(),e.preventDefault())},onKeyUp:e=>{e.key===" "&&(e.stopPropagation(),e.preventDefault(),n())}}}function Sl(n){return{onKeyDown:e=>{e.key==="Enter"&&(e.stopPropagation(),e.preventDefault(),n())}}}function kl(n,e){return{onKeyDown:t=>{t.key==="ArrowUp"&&(t.stopPropagation(),t.preventDefault(),n()),t.key==="ArrowDown"&&(t.stopPropagation(),t.preventDefault(),e())}}}const Al=["cancel","submit","advanced_pref","accept_all","decline_all","ask_me_later","show_pref_mgr","yes","no","back","close"],qe=({buttons:n=[],className:e=""})=>{const t=R({}),i=R({}),o=y(h=>h.configSlice.config),a=y(h=>h.globalStateSlice.currentView),r=Zn(),l=y(h=>h.globalStateSlice.advRootProps),c=y(h=>h.globalStateSlice.shouldShowDisabledSubmitMessage),[d,p]=_n(!1);W(()=>{p(!0)},[]);const u=j(l),f=V(o.contents.configuration_accessible_layout_key)||u;function g(){let h=e;return h||(j(l)?a==="euDisclosurePanel"?h=n.filter(_=>{var k;return!((k=_.inlineStyle)!=null&&k.includes("display: none"))}).length===3?"disclosureButtonPanel buttonPanelNoWrap":"disclosureButtonPanel":Nn(l)?h="prefTableButtonPanel":fn(l)&&a==="gppPanel"?h="bottom":h="accessibleDynamicButtons":h="pdynamicbutton"),r&&(h+=" gsSubmitContainer"),h}const x=h=>m=>{t.current[h]=m},b=h=>m=>{i.current[h]=m};return s("div",{className:g(),style:"visibility:visible",id:u&&a!=="optoutDoneMessage"?"accessibleDynamicButtons":void 0,children:[(a==="advancedPanel"&&Bn(l)||a==="euDisclosurePanel"&&Pn(l))&&s("div",{className:u?"accessibleGsOptedOutMsg":""}),u&&c&&s(Cl,{}),n.map(({key:h,id:m,label:_,onClick:k,onMouseEnter:A,onMouseLeave:S,styleName:M="",inlineStyle:z,disabled:O=!1},F)=>{let N=He(_);if(fn(l)){const G=hn();h==="submit"?N=`<img src="${G}asset/check.svg" alt="">${N}`:h==="accept_all"?N=`<img src="${G}asset/thumbs-up.svg" alt="">${N}`:h==="decline_all"&&(N=`<img src="${G}asset/thumbs-down.svg" alt="">${N}`)}let B=f;const C=`${h}-${F}-${a}`;j(l)?B=!0:a==="euDisclosurePanel"&&h==="show_pref_mgr"&&(B=V(o.contents.disclosure_is_show_pref_mgr_button_key));const D=h+"Button";return B?s("button",{id:m,className:M,style:z,onClick:k,...A&&{onMouseEnter:A},...S&&{onMouseLeave:S},disabled:O,ref:b(C),children:d?s(U,{containerRef:{current:i.current[C]},html:N}):null},D):s("a",{id:m,href:"#",className:M,style:z,onClick:G=>{G.preventDefault(),k()},...qn(k),tabIndex:0,role:"button",...A&&{onMouseEnter:A},...S&&{onMouseLeave:S},ref:x(C),children:[d?s(U,{containerRef:{current:t.current[C]},html:N}):null,s(U,{containerRef:{current:t.current[h]},html:N})]},D)})]})};function Se(){const n=rn(),e=y(v=>v.configSlice.config),t=y(v=>v.globalStateSlice.advRootProps),i=y(v=>v.preferencesSlice.activeCategories),o=y(v=>v.preferencesSlice.activeVendors),a=y(v=>v.preferencesSlice.activeGppPurposes),r=y(v=>v.preferencesSlice.activeGppSubsections),l=y(v=>v.globalStateSlice.currentView),c=y(v=>v.globalStateSlice.previousView),d=y(v=>v.globalStateSlice.startingView),p=qt(),u=Oo(),f=y(v=>v.globalStateSlice.shouldShowDisabledSubmitMessage),g=Zn(),x=R(null),b=e.config.gppCountryStates&&e.config.gppCountryStates.length>0?e.config.gppCountryStates[0]:void 0,h=j(t),m=async()=>{await(u==null?void 0:u.doConsent(yn.SUBMIT_PREFERENCES,{activeCategories:i,activeVendors:o})),fn(t)&&b&&Dr({gppCountryState:b,activeGppPurposes:a,activeGppSubsections:r,categories:e.categories,activeCategories:i}),de("truste-click-button-submit",{})},_=async()=>{const v=Ce({level:0,allCategories:e.categories,dispatch:n});await(u==null?void 0:u.doConsent(yn.DECLINE_ALL,v)),fn(t)&&b&&Nr()},k=()=>{const v=l==="privacyPolicyPanel";if(v&&c){n(un(c));return}l===d||v&&!c?En():n(un(d))},A=()=>{n(un(l==="sliderPanel"?"advancedPanel":"sliderPanel"))},S=async()=>{const v=p.logInformationalActivity(Me.ASK_ME_LATER).catch(q=>{T.error("Failed to log ask-me-later activity: {}",q)});await Lo(v,mo),En()},M=()=>{p.logInformationalActivity(Me.MORE_INFORMATION),n(un(ze(t,e.contents)))},z=Vt().toLowerCase()===el,O=l===d||l==="privacyPolicyPanel"&&!c,F=!Cr(),N=O&&z&&F,B={key:"cancel",label:(l==="privacyPolicyPanel"?e.contents.privacypolicypanel_back_key:e.contents.preftable_cancelbutton_label_key)||$s,onClick:k,styleName:Nn(t)?"ta-modal-secondary-btn prefTableCancel accessibleButton":"cancel nocon"},C=Object.values(i),D=l==="advancedPanel"&&e.categories.length>1&&(C.length===0||C.includes(null));W(()=>{h&&l==="advancedPanel"?n(D?et():tt()):h&&n(tt())},[l,D,h,i]);const G={key:"submit",label:e.contents.preftable_submitbutton_label_key||zs,onClick:D?()=>{h&&n(et())}:m,styleName:J(),disabled:h&&D,onMouseEnter:()=>{D&&!h&&n(et())},onMouseLeave:()=>{f&&!h&&n(tt())}};function J(){let v;return Nn(t)?v="ta-modal-primary-btn prefTableSubmitFilled accessibleButton":fn(t)?v="ta-modal-secondary-btn submit":h?v="ta-modal-primary-btn accessibleButtonPrimary submitPreferencesButtonLower":(v="submit",g&&Bn(t)&&(v+=" gsSubmit")),D&&(v+=" disable"),v}const Z=l==="advancedPanel"&&V(e.contents.preftable_show_accept_all_button_lower_key),L=l==="euDisclosurePanel",on=l==="euDisclosurePanel"?e.contents.disclosure_consenttoall_key:e.contents.preftable_accept_all_button_lower_text_key;function Ln(){return L?h?"ta-modal-primary-btn disclosureAcceptAll accessibleButton":"call":Nn(t)?"ta-modal-primary-btn prefTableAccept accessibleButton":h?"ta-modal-primary-btn accessibleButtonPrimary acceptAllButtonLower":"acceptAllButtonLower"}const Hn=Ln(),jn=Z||L||fn(t)?{key:"accept_all",label:on||vo,onClick:()=>Ct({currentView:l,cmConfig:e,rootProps:t,consent:u,dispatch:n}),styleName:Hn}:void 0,On=l==="euDisclosurePanel"&&V(e.contents.disclosure_showrequiredonly_key),Sn=l==="advancedPanel"&&V(e.contents.preftable_show_decline_all_button_lower_key),I=L?e.contents.disclosure_requiredonly_key:e.contents.preftable_decline_all_button_lower_text_key;function gn(){return L?h?"ta-modal-primary-btn disclosureDeclineAll accessibleButton":"required":Nn(t)?"ta-modal-primary-btn prefTableDecline accessibleButton":h?"ta-modal-primary-btn accessibleButtonPrimary declineAllButtonLower":"declineAllButtonLower"}const wn=gn(),kn={key:"decline_all",label:I||Co,onClick:_,styleName:wn,inlineStyle:On||Sn||fn(t)?"":"display: none;"},zn=V(e.contents.preftable_show_advance_settings_value_key??"true");function w(){return Nn(t)?"ta-modal-secondary-btn prefTableSettings accessibleButton":"advance floatRight noMarginRight"}const P={key:"advanced_pref",label:l==="sliderPanel"?e.contents.preftable_advancesetting_label_key??Ws:e.contents.preftable_basicsetting_label_key??js,onClick:A,styleName:w()},Q=z&&V(e.contents.disclosure_showaskmelaterbutton_key)?{key:"ask_me_later",label:e.contents.disclosure_askmelater_key||Us,onClick:S,styleName:h?"ta-modal-secondary-btn disclosureAskMeLater accessibleButton":"askmelater"}:void 0;function an(){return L&&h?"ta-modal-secondary-btn disclosureViewCookieSettings accessibleButton":Nn(t)?"ta-modal-secondary-btn prefTableSettings accessibleButton":"shp"}const cn={key:"show_pref_mgr",label:e.contents.disclosure_showpreferencemanager_key||Vs,onClick:M,styleName:an()},Dn={cancel:N?void 0:B,submit:G,accept_all:jn,decline_all:kn,...zn&&P.label&&{advanced_pref:P},ask_me_later:Q,show_pref_mgr:cn};function Fn(v,q){return v.split(",").map(en=>en.trim()).filter(en=>{const sn=Al.includes(en);return sn||T.warn("[getOrderedButtons] Invalid button key: {}",en),sn}).map(en=>q[en]).filter(en=>!!en)}let Xn;if(l==="euDisclosurePanel")Xn=e.contents.disclosure_button_order_key||qs;else if(l==="privacyPolicyPanel")Xn=Qs;else{const v=Bn(t)||fn(t)?Ks:Ys;Xn=e.contents.preftable_button_order_key||v}const ke=Fn(Xn,Dn);return s(E,{children:[Pn(t)&&(l==="advancedPanel"||l==="sliderPanel")&&s("div",{ref:x,children:e.contents.preftable_pre_button_panel_key&&s(U,{containerRef:x,html:e.contents.preftable_pre_button_panel_key})}),s(qe,{buttons:ke})]},"action-buttons")}function Ke(){const n=rn();return ln(e=>{e.preventDefault(),n(un("advancedPanel"))},[n])}function Qe(){const n=rn(),e=y(t=>ze(t.globalStateSlice.advRootProps,t.configSlice.config.contents));return ln(t=>{t.preventDefault(),n(un(e))},[n,e])}function Rn(n){const e=y(i=>i.configSlice.config),t=y(i=>i.globalStateSlice.advRootProps);return wo(n,t,e==null?void 0:e.config)}function Fo(){const n=y(t=>t.configSlice.config),e=y(t=>t.globalStateSlice.advRootProps);return t=>wo(t,e,n==null?void 0:n.config)}function fe({forGranularScroll:n=!1}){const e=y(h=>h.globalStateSlice.advRootProps),t=y(h=>h.configSlice.config.contents),i=y(h=>h.globalStateSlice.currentView),o=R(null),a=Zn(),r=(e==null?void 0:e.gpchonored)==="true",l=(t==null?void 0:t.preftable_gpc_recognition_notice_key)==="true",c=Rn((t==null?void 0:t.preftable_gpc_recognition_message_key)||'<img src="asset/gpc-check.svg" alt="" aria-hidden="true" />GPC opt-out preference honored'),d=j(e),u=["euDisclosurePanel","sliderPanel","advancedPanel","gppPanel"].includes(i),f=a&&i==="advancedPanel";return r&&l&&u&&(n?f:!f)?s(E,{children:[s("div",{className:n?"trustarc-gpc-notice-gs":d?"trustarc-gpc-notice pSemiBold":"trustarc-gpc-notice",ref:o,children:s(U,{containerRef:o,html:c})}),i==="euDisclosurePanel"&&s("br",{})]}):null}function Tl(){const n=rn(),e=y(g=>g.configSlice.config),t=y(g=>g.globalStateSlice.advRootProps),i=Vt(),{createEventHandlers:o}=We(),a=R(null),r=j(t),l=Fo(),c=Qe(),d=Ke(),p=$(()=>o({"#privacyPolicy":{click:()=>n(un("privacyPolicyPanel"))},"#cookieSettingsLink":{click:c},"#advancedSettingsLink":{click:d}}),[n,o,c,d]),u=$(()=>{const g=e.contents.disclosure_consent_key,x=e.contents.disclosure_ask_me_later_message_key,b=g==null?void 0:g.replace("${ASKMELATER_MESSAGE}",i==="expressed"?x:"");return l(b)},[e,i,l]);return s(E,{children:[s("div",{...r?{className:"disclosureBody",id:"cookieConsentDescription"}:{key:"eu-disclosure-panel",id:"cookieConsentDescription",className:"consent editable"},ref:a,children:s(U,{containerRef:a,html:u,eventHandlers:p})}),s(fe,{}),s(Se,{}),!r&&s(E,{children:[s("br",{className:"spacing"}),s("br",{className:"spacing br-clear"})]})]})}function Pl(){return s("div",{className:"gpp_loading_container",children:s("img",{src:`${hn()}asset/spinner.svg`,className:"loadingSpinner","aria-hidden":"true",alt:""})})}function Ze({toggleState:n,onLabel:e,offLabel:t,onToggle:i,ariaLabelOn:o,ariaLabelOff:a,isVisible:r=!0,removeSwitchParentRole:l,removeSwitchParentAriaLabel:c,isAccessibleSwitch:d,isVendor:p,isGppPurposeSwitch:u,switchAriaLabel:f}){const g=y(O=>O.globalStateSlice.advRootProps),x=y(O=>O.configSlice.gppLoading),b=n===!0,h=n===!1,m=R(null),_=R(null),k=R(null),A=R(null),S=d?"button":"div";if(u&&x)return s(Pl,{});function M(){return u?"accessibleSwitch gpp_switch":j(g)?"accessibleSwitch":"switch"}const z=({ref:O,isActive:F,positionClass:N,ariaLabel:B,ariaPressed:C,toggleValue:D,containerRef:G,html:J})=>s("span",{ref:O,className:`gwt-InlineHTML ${N} ${F?"active":""}`,role:d?void 0:"button",tabIndex:d?void 0:0,"aria-label":B,"aria-disabled":"false","aria-pressed":d?void 0:C,style:p?void 0:{visibility:r?"visible":"hidden"},...!d&&{onClick:()=>i(D),...qn(()=>i(D))},children:s(U,{containerRef:G,html:J})});if(j(g)){if(d){const F=s("div",{className:"truste-switch-wrapper",children:[s("span",{ref:_,className:"truste-switch-label",children:s(U,{containerRef:_,html:b?e:t})}),s("label",{className:"truste-switch","aria-label":f,children:[s("input",{type:"checkbox",checked:b,onChange:()=>i(!b),...Sl(()=>i(!b))}),s("span",{className:"truste-slider round"})]})]});return p?s("div",{className:"tableSwitchPanelContainer",children:F}):F}const O=s("div",{className:M(),role:"radiogroup",children:[s("button",{ref:A,type:"button",className:`gwt-Button on${h?" active":""}`,role:"radio",tabIndex:0,"aria-label":a,"aria-disabled":"false","aria-checked":h,style:"visibility: visible;",onClick:()=>i(!1),children:s(U,{containerRef:A,html:t})}),s("button",{ref:k,type:"button",className:`gwt-Button off${b?" active":""}`,role:"radio",tabIndex:0,"aria-label":o,"aria-disabled":"false","aria-checked":b,style:"visibility: visible;",onClick:()=>i(!0),children:s(U,{containerRef:k,html:e})})]});return p?s("div",{className:"tableSwitchPanelContainer",children:O}):O}return s(S,{className:M(),role:l?void 0:d?"switch":"listbox","aria-label":c?void 0:f??"Switch","aria-checked":d&&n!==null?b:void 0,...d&&{onClick:()=>i(!b)},children:[z({ref:_,isActive:h,positionClass:"on",ariaLabel:a,ariaPressed:h,toggleValue:!1,containerRef:_,html:t}),z({ref:m,isActive:b,positionClass:"off",ariaLabel:o,ariaPressed:b,toggleValue:!0,containerRef:m,html:e})]})}function El({domains:n,companyName:e,triggerLabel:t="Show All",isVisible:i=!1,onToggle:o}){const a=R(`${e}-${crypto.randomUUID()}`).current,r=()=>{o==null||o()},l=()=>{o==null||o()},c=i?"accessibleShowAll tooltipActivelyShown":"accessibleShowAll",d=i?{display:"flex"}:{display:"none"};return s(E,{children:[s("a",{className:c,href:"javascript:;","aria-label":`${t} ${e} domains`,role:"button","aria-hidden":"false",tabIndex:0,onClick:p=>{p.preventDefault(),p.stopPropagation(),r()},children:t}),s("div",{className:"gwt-HTML accessibleTooltip pSemiBold showAllTooltipDown",id:a,style:d,children:[s("div",{className:"gwt-HTML tooltipHeader",children:[s("p",{className:"tooltipHeaderText",children:s("strong",{children:["All domains of ",e]})}),s("button",{type:"button",className:"gwt-Button ta-modal-secondary-btn ta-modal-btn-small","aria-label":"Close Tooltip",onClick:l,children:s("img",{src:`${hn()}asset/close-button.svg`,className:"gwt-Image","aria-hidden":"true"})})]}),s("ul",{children:n.map((p,u)=>s("li",{children:p},u))})]})]})}function No({vendor:n,index:e,categoryName:t,categoryId:i,toggleState:o,isRequired:a,isGppContext:r=!1}){const l=rn(),c=xn(i,n.id),d=y(I=>I.configSlice.config),p=y(I=>I.globalStateSlice.advRootProps),u=R(null),[f,g]=_n(!1),[x,b]=_n(!1),h=y(I=>I.globalStateSlice.isMobile),m=$(()=>n.displayname||n.name,[n]),_=y(I=>I.preferencesSlice.activeVendors[c]!==void 0?I.preferencesSlice.activeVendors[c]:o),k=y(I=>!!I.preferencesSlice.expandedVendors[c]),A=y(I=>!!I.preferencesSlice.showAllDomainsVendors[c]),M=y(I=>I.preferencesSlice.activeTooltipVendor)===c,z=n.domains.join(", "),O=`vendorDetails-${c}`,F=$(()=>V(d.contents.preftable_enable_aria_disclosure_controls_key)?O:void 0,[d.contents.preftable_enable_aria_disclosure_controls_key,O]),N=d.contents.preftable_out_label_key??Eo,B=d.contents.preftable_in_label_key??Po;W(()=>{const I=u.current;if(!I)return;const gn=I.scrollWidth>I.offsetWidth;g(gn),Pn(p)&&!L?b(!0):b(gn)},[z,p]);const C=()=>{l(Kr({categoryId:i,vendorId:n.id}))},D=I=>{var gn,wn;(gn=I==null?void 0:I.preventDefault)==null||gn.call(I),(wn=I==null?void 0:I.stopPropagation)==null||wn.call(I),l(Qr({categoryId:i,vendorId:n.id}))},G=()=>{l(Zr(M?null:{categoryId:i,vendorId:n.id}))},J=I=>{l(qr({categoryId:i,vendorId:n.id,vendorToggledOn:I,allCategories:d.categories}))},Z=(I,gn,wn,kn)=>{const w=`${I.replace("{CATEGORY_NAME_DO_NOT_REMOVE}",gn)} ${wn}`;return V(d.contents.preftable_enable_in_out_aria_labels_key)&&!w.includes(kn)?`${kn} - ${w}`:w},L=j(p),on=Pn(p),Ln=L?"gdprTableAccordion":void 0,Hn=!A,jn=L?"domainPanelContainer":"",On=Hn?`hideText${L?" accessibleDomainText":""}`:"",Sn=`${f?"":"hiddenSeeMore"}${L?" accessibleShowAll":""}`;return s(E,{children:[s("tr",{className:e%2===1?"even":"",children:[s("td",{className:Ln,children:L?s("button",{type:"button",className:`ta-modal-tertiary-btn tc-icon ${k?"tc-icon-table-chevron-down":"tc-icon-table-chevron-up"}`,"aria-expanded":k?"true":"false",tabIndex:0,"aria-hidden":"false","aria-label":`Learn more about ${m}`,"aria-controls":F,onClick:C}):s("img",{className:`collapse ${k?"colminus":"colplus"}`,role:"button",tabIndex:0,"aria-hidden":"false","aria-label":`${k?d.contents.preftable_hide_table_row_icon_aria_label_key||ol:d.contents.preftable_show_table_row_icon_aria_label_key||al}`,"aria-controls":F,"aria-expanded":k?"true":"false",onClick:C,...qn(C)})}),s("td",{children:m}),s("td",{children:s("div",{className:jn,children:[h?n.domains.length>1?s("a",{href:"javascript:;","aria-hidden":"false",tabIndex:0,role:"button",onClick:I=>D(I),...qn(()=>D()),children:A?d.contents.preftable_hide_label_key||vi:d.contents.preftable_show_all_label_key||at}):s("div",{children:z}):s(E,{children:[s("div",{ref:u,className:On,children:z}),x&&L&&s(El,{domains:n.domains,companyName:m,triggerLabel:d.contents.preftable_show_all_label_key||at,isVisible:M,onToggle:G})]}),!h&&x&&!L&&s("a",{className:Sn,href:"javascript:;","aria-hidden":"false",tabIndex:0,role:"button",onClick:I=>D(I),...qn(()=>D()),children:A?d.contents.preftable_hide_label_key||vi:d.contents.preftable_show_all_label_key||at})]})}),!a&&!r&&s("td",{children:s(Ze,{toggleState:_,offLabel:N,onLabel:B,onToggle:J,ariaLabelOff:Z(d.contents.preftable_company_switch_aria_label_out_key||tl,t,m,N),ariaLabelOn:Z(d.contents.preftable_company_switch_aria_label_in_key||il,t,m,B),removeSwitchParentRole:V(d.contents.preftable_remove_switch_parent_role_key),removeSwitchParentAriaLabel:V(d.contents.preftable_remove_switch_parent_aria_label_key),isAccessibleSwitch:V(d.contents.preftable_accessible_switch_key),isVendor:!0,switchAriaLabel:`${m} ${(d.contents.preftable_choice_label_key||"Choice").toLowerCase()}`})})]}),h&&A&&n.domains.length>1&&s("tr",{children:s("td",{colSpan:100,children:s("div",{className:"mobileDomainCollapse",children:`${d.contents.preftable_domains_label_key||Js} ${z}`})})}),s("tr",{id:O,style:{display:k?"table-row":"none"},"aria-hidden":!k,children:s("td",{colSpan:100,children:s("div",{className:"whyCantIOptOut",children:[L&&s("h4",{className:"gwt-Label pbold",children:d.contents.preftable_description_label_key??"Description"}),s("div",{className:"gwt-HTML",tabIndex:L?void 0:0,dangerouslySetInnerHTML:{__html:n.description||d.contents.preftable_under_review_text_key||pl}}),L&&(n.tacUrl||n.websitePrivacyPolicy)&&s("div",{className:"linksContainer",children:[n.tacUrl&&s("a",{className:"gwt-Anchor companyLinks",href:n.tacUrl,target:"_blank",rel:"noopener",children:d.contents.preftable_terms_and_conditions_label_key||cl}),n.tacUrl&&n.websitePrivacyPolicy&&s("div",{className:"dividerContainer",children:s("div",{className:"dividerContent"})}),n.websitePrivacyPolicy&&s("a",{className:"gwt-Anchor companyLinks",href:n.websitePrivacyPolicy,target:"_blank",rel:"noopener",children:d.contents.preftable_privacy_policy_label_key||rt})]}),!L&&n.websitePrivacyPolicy&&s("a",{id:on?"privacyPolicyAnchor":`privacyPolicyAnchor-${m}-${t}`,className:on?"gwt-Anchor":"privacy-policy-anchor",href:n.websitePrivacyPolicy,target:"_blank",rel:"noopener","aria-label":`${m} ${d.contents.preftable_privacy_policy_label_key||rt}`,children:d.contents.preftable_privacy_policy_label_key||rt})]})})})]})}const Ll=({children:n,isNonAccessibleDefaultEu:e,containerClass:t,containerStyle:i,isExpanded:o})=>e?s(E,{children:n}):s("div",{className:t,style:i,"aria-hidden":!o,children:n});function Ol({categoryWithIndex:{category:n,index:e},categoryId:t,isExpanded:i,toggleState:o,onToggleExpand:a,onToggleActive:r}){const l=o===!0,c=o===!1,d=y(v=>v.globalStateSlice.advRootProps),p=y(v=>v.configSlice.config),u=R(null),f=R(null),g=R(null),x=R(null),b=R(null),h=R(null),m=j(d),_=p.contents.preftable_active_label_key||dl,k=$(()=>Bn(d),[d]),A=$(()=>fn(d),[d]),S=$(()=>Nn(d),[d]),M=$(()=>Pn(d),[d]),z=!m&&M,O=$(()=>Vr(n),[n]),F=$(()=>`${n.displayText}. ${ue(n.description)}`.trim(),[n]),N=$(()=>!(k||A)&&V(p.contents.preftable_enable_group_tab_index_aria_label_prefpanel_key,!0)&&!m,[k,A,p.contents.preftable_enable_group_tab_index_aria_label_prefpanel_key,m]),B=$(()=>p.contents.preftable_required_switch_key==="true",[p.contents.preftable_required_switch_key]),C=$(()=>e===0&&!B,[e,B]),D=$(()=>p.contents.configuration_accessible_layout_key==="true"||j(d),[d.layout,p.contents.configuration_accessible_layout_key]),G=$(()=>p.contents.preftable_accessible_switch_key==="true",[p.contents.preftable_accessible_switch_key]),J=$(()=>{const v=V(p.contents.preftable_radio_button_switch_key);return k?!G&&v:v},[G,p.contents.preftable_radio_button_switch_key,d]),Z=$(()=>Aa.forwardRef((v,q)=>s(D?"h2":"h3",{ref:q,...v})),[D]),L=$(()=>p.contents.preftable_example_purposes_label_key||ul,[p.contents]),on=$(()=>(p.contents.preftable_category_panel_order_key||rl).split(","),[p.contents]),Ln=$(()=>V(p.contents.preftable_enable_aria_disclosure_controls_key)?"cookieInfoButton":void 0,[p.contents.preftable_enable_aria_disclosure_controls_key]),Hn=ln(()=>{var vn;const v=i?"preftable_hide_cookie_key":"preftable_view_cookie_key";let q;j(d)?q="See more detailed preferences":i?q="Hide Cookies":q="View Cookies";const tn=p.contents[v];if(tn==null)return q;const en=tn.split("||");let sn;return k&&en.length>1&&e<en.length?sn=e:sn=0,(vn=en[sn])==null?void 0:vn.trim()},[p.contents,i]),jn=ln(()=>{const v=i?"preftable_hide_cookie_aria_label_key":"preftable_view_cookie_aria_label_key",q=i?"Hide":"View";let tn=p.contents[v]||q;return tn.includes("{CATEGORY_NAME}")&&(tn=tn.replace("{CATEGORY_NAME}",O)),tn},[p.contents,O,i]),On=ln(()=>{const v=p.contents.preftable_out_label_key,q=p.contents.preftable_in_label_key,tn=p.contents.preftable_out_aria_label_key||"Refuse",en=p.contents.preftable_in_aria_label_key||"Accept";return{outLabel:v??(J?"Off":Eo),inLabel:q??(J?"On":Po),outAriaLabel:`${tn} ${O}`,inAriaLabel:`${en} ${O}`}},[p.contents,O,J]),Sn=Rn(n.description),I=ln(()=>{let v=s(E,{children:[s("span",{className:"gwt-InlineHTML",children:s(Z,{ref:u,children:s(U,{containerRef:u,html:n.displayText})})}),s("span",{className:"gwt-InlineHTML",dangerouslySetInnerHTML:{__html:Sn}})]});return D&&(v=s(E,{children:[s("div",{className:m?"gwt-HTML accessibleCategoryHeadingName":"gwt-HTML",children:s(Z,{ref:u,children:s(U,{containerRef:u,html:n.displayText})})}),s("div",{className:"gwt-HTML",children:s("p",{dangerouslySetInnerHTML:{__html:Sn}})})]})),!N&&!m?s("div",{tabIndex:0,children:v}):v},[n,O,D,N]),gn=G?"button":"div",wn=ln(()=>{const{outLabel:v,inLabel:q}=On(),tn=`${O} Preferences`,en=`gwt-uid-${n.id}-on`,sn=`gwt-uid-${n.id}-off`,vn=()=>{r(!l)};return s(gn,{className:"switch",role:V(p.contents.preftable_remove_switch_parent_role_key)?void 0:G?"switch":"radiogroup","aria-label":V(p.contents.preftable_remove_switch_parent_aria_label_key)?void 0:"Switch","aria-checked":G?l:void 0,disabled:G?C:void 0,...G&&{onClick:vn},children:[s("span",{className:`gwt-RadioButton on ${c?"active":""}`,"aria-checked":c,"aria-disabled":C?"true":"false","aria-pressed":c,style:{visibility:"visible"},children:[s("input",{type:"radio",name:tn,value:"on",id:sn,tabIndex:0,checked:c,onChange:()=>r(!1),disabled:C}),s("label",{htmlFor:sn,ref:f,children:s(U,{containerRef:f,html:v})})]}),s("span",{className:`gwt-RadioButton off ${l?"active":""}`,"aria-checked":l,"aria-disabled":"false","aria-pressed":l,style:{visibility:"visible"},children:[s("input",{type:"radio",name:tn,value:"on",id:en,tabIndex:0,checked:l,onChange:()=>r(!0)}),s("label",{htmlFor:en,ref:g,children:s(U,{containerRef:g,html:q})})]})]})},[O,n.id,o,r,On,G,l,c,p,C]),kn=He(Hn()??""),zn=jn(),w=On(),P=()=>{if(m)return i?"accordion active":"accordion";let v="viewcookie";return i&&(k?v="viewcookie active":v="expandcookie"),v},K=()=>s(E,{children:D?s("button",{ref:x,className:P(),"aria-label":zn,"aria-expanded":i?"true":"false","aria-controls":Ln,style:{visibility:"visible"},onClick:a,children:s(U,{containerRef:x,html:kn??""})}):s("a",{ref:b,className:P(),href:"javascript:;",role:"button",tabIndex:0,"aria-label":zn,"aria-expanded":i?"true":"false","aria-controls":Ln,style:{visibility:"visible"},onClick:v=>{v.preventDefault(),v.stopPropagation(),a()},...qn(a),children:s(U,{containerRef:b,html:kn??""})})}),Q=()=>{const v=i?"chevron-down-grey.svg":"chevron-right-grey.svg";return s("button",{type:"button",className:P(),"aria-label":zn,"aria-expanded":i?"true":"false",style:{pointerEvents:"auto",visibility:"visible"},onClick:a,children:[s("img",{src:`${hn()}asset/${v}`,className:"",alt:""}),s("div",{className:"gwt-HTML",ref:h,children:s(U,{containerRef:h,html:kn??""})})]})},an=()=>s(E,{children:[!C&&(J?wn():s(Ze,{toggleState:o,offLabel:w.outLabel,onLabel:w.inLabel,onToggle:r,ariaLabelOff:w.outAriaLabel,ariaLabelOn:w.inAriaLabel,removeSwitchParentRole:V(p.contents.preftable_remove_switch_parent_role_key),removeSwitchParentAriaLabel:V(p.contents.preftable_remove_switch_parent_aria_label_key),isAccessibleSwitch:G,isVendor:!1,switchAriaLabel:`${O} ${(p.contents.preftable_choice_label_key||"Choice").toLowerCase()}`})),C&&!m&&s("div",{"aria-hidden":"true"})]}),cn=(v,q)=>{const tn=V(p.contents.preftable_use_non_header_cell_on_empty_header_key)&&!v,en=tn?"td":"th",sn={};return q&&(sn["aria-label"]=q),tn||(sn.scope="col"),s(en,{...sn,children:ue(v)})},Dn=()=>{var sn;const v=m?"accordionContentContainer":"",q=i?{display:"block"}:{display:"none"},tn=i?void 0:{display:"none"},en=()=>m?n.vendors.length===0?"p-semibold noCookiesTable gdprAccessibleTable":`${S?"prefTableTable":"gdprAccessibleTable"} ${C?"optoutless":"withoptout"}`:`prefTable ${C?"optoutless":i&&n.vendors.length>0?"withoptout":""}`;return s(E,{children:s(Ll,{isNonAccessibleDefaultEu:z,containerClass:v,containerStyle:q,isExpanded:i,children:[((sn=n.details)==null?void 0:sn.length)>0&&(k||A||S)&&s("div",{className:"prefPanel",children:m?s(E,{children:[s("div",{className:"examplePurposes pbold",children:s("div",{className:"gwt-HTML",children:s("p",{children:L})})}),s("div",{className:"categoryDetailsList",children:n.details.map((vn,ae)=>s("div",{className:"gwt-HTML",children:s("p",{dangerouslySetInnerHTML:{__html:vn.text}})},`detail-${ae}`))})]}):s(E,{children:[s("div",{className:"left",children:s("span",{className:"gwt-InlineHTML",children:L})}),s("div",{className:"right",children:n.details.map((vn,ae)=>s("span",{className:"gwt-InlineHTML",dangerouslySetInnerHTML:{__html:vn.text}},`detail-${ae}`))})]})}),k&&s("span",{className:"gwt-InlineHTML",children:s("br",{"aria-hidden":"true",className:"spacing br-clear"})}),s("table",{className:en(),...m&&{init:"true"},...z&&{style:tn},children:[i&&s(E,{children:[s("colgroup",{children:n.vendors.length>0?m?C?s(E,{children:[s("col",{width:"56px",className:"prefTableRequiredArrowColumn"}),s("col",{width:"31%",className:"prefTableRequiredCompanyColumn"}),s("col",{width:"63%",className:"prefTableRequiredDomainsColumn"})]}):s(E,{children:[s("col",{width:"56px",className:"prefTableArrowColumn"}),s("col",{width:"24%",className:"prefTableCompanyColumn"}),s("col",{width:"39%",className:"prefTableDomainsColumn"}),s("col",{width:"32%",className:"prefTableSwitchColumn"})]}):s(E,{children:[s("col",{width:"2%"}),C?s(E,{children:[s("col",{width:"28%"}),s("col",{width:"70%"})]}):s(E,{children:[s("col",{width:"18%"}),s("col",{width:"45%"}),s("col",{width:"30%"})]})]}):s("col",{width:"100%"})}),s("thead",{children:s("tr",{children:n.vendors.length>0?s(E,{children:[cn(p.contents.preftable_table_first_column_first_row_label_key||"",p.contents.preftable_table_first_column_first_row_aria_label_key),cn(p.contents.preftable_company_label_key||Zs),cn(p.contents.preftable_domain_label_key||Xs),!C&&cn(p.contents.preftable_optout_label_key??nl)]}):cn(p.contents.preftable_cookies_dont_exist_label_key||To)})})]}),!i&&s("colgroup",{children:[s("col",{width:"2%"}),s("col",{width:"28%"}),s("col",{width:"70%"})]}),s("tbody",{children:i&&n.vendors.map((vn,ae)=>s(No,{vendor:vn,index:ae,categoryName:O,categoryId:t,toggleState:o,isRequired:C},`vendor-${vn.id}`))})]})]})})},Fn={view_cookie:m?Q:K,category_switch:an,pref_panel:Dn};function Xn(){return m?"categoryPanelAccessible":"cookiecat"}const ke=s("div",{className:Xn(),children:m?s(E,{children:[s("div",{className:"catDivContainer",children:[s("div",{className:"categoryDivContent",children:I()}),C&&s("div",{className:"required-active",style:{display:"block"},children:s("div",{className:"gwt-HTML",children:s("p",{children:_})})}),an()]}),Q(),Dn()]}):s(E,{children:[I(),on.map(v=>{const q=Fn[v];return(q==null?void 0:q())??null})]})});return m?ke:s("div",{role:N?"group":void 0,tabIndex:N?0:void 0,"aria-label":N?F:void 0,children:ke})}function Io(){const n=rn(),e=y(b=>b.globalStateSlice.advRootProps),t=y(b=>b.configSlice.config),i=y(b=>b.preferencesSlice.expandedCategories),o=y(b=>b.preferencesSlice.activeCategories),a=y(b=>b.uiStateSlice.titleLogoExist),r=y(b=>b.uiStateSlice.headerDescriptionElements),l=Zn(),c=y(b=>b.globalStateSlice.currentView),d=R([]),p=fn(e);d.current=r.map((b,h)=>d.current[h]||pe());const u=b=>{n(Wr(b))},f=(b,h,m)=>{$r({categoryId:b,categoryIndex:h,targetState:m,allCategories:t.categories,dispatch:n})};function g(){let b="prefPanel";return j(e)&&(p?b+=" gppPrefPanel":b+=" gdprAccessiblePrefPanel"),l&&(a||(b+=" gsPrefPanelNoLogo"),b+=" gsPrefPanel"),p&&(b+=" prefTableAdvanced"),b}function x(){if(l)return`${t.contents.preftable_granular_scroll_custom_height_key??sl}px`}return s(E,{children:[c==="advancedPanel"&&!j(e)&&s("div",{"aria-hidden":"true",style:"display: none;"}),j(e)&&s("div",{className:"divider",children:s("hr",{})}),s("div",{className:g(),style:{visibility:"visible",transition:"all",maxHeight:x()},children:[r.map(({id:b,html:h},m)=>{const _=d.current[m];return s("div",{className:"gwt-HTML",ref:_,children:s(U,{containerRef:_,html:h})},`desc-${m}-${b}`)}),l&&s(fe,{forGranularScroll:!0}),t.categories.map((b,h)=>({category:b,index:h})).filter(({category:b})=>{var k,A;const h=t.contents.uncategorized_label,m=!!h&&b.displayText===h||((k=b.defaultLocaleName)==null?void 0:k.toLowerCase().includes("uncategorized"))||((A=b.name)==null?void 0:A.toLowerCase().includes("uncategorized")),_=!b.vendors||b.vendors.length===0;return!(m&&_)}).map(({category:b,index:h},m,_)=>{const k=b.id,A=m===_.length-1;return s(E,{children:[s(Ol,{categoryWithIndex:{category:b,index:h},categoryId:b.id,isExpanded:!!i[k],toggleState:o[k],onToggleExpand:()=>u(k),onToggleActive:S=>f(k,h,S)},`category-${b.id}`),j(e)&&!A&&s("div",{className:"divider",children:s("hr",{})})]},`category-fragment-${b.id}`)})]}),!Bn(e)&&s("br",{"aria-hidden":"true",className:"spacing br-clear"}),!fn(e)&&s(Se,{})]})}function lt(n,e){T.error("[CCM] Closing modal due to error: {}",n),e(Ie()),e(xt()),setTimeout(()=>{En()},5e3)}const Dl=n=>{if(!n.contents)return n;const e=globalThis.location.protocol,t=Object.entries(n.contents).reduce((i,[o,a])=>(i[o]=typeof a=="string"?ue(a.split("{HTTP_PROTOCOL}").join(e).split("{URL_PROTOCOL}").join(e)):a,i),{...n.contents});return{...n,contents:t}},Ro=(n,e)=>{const{rootProps:t,effectiveLocale:i}=e,o=t.cmType==="PRO",a=p=>{const u=p.categories;return p.categoriesWithIab=u,p.categories=Hr(u),p},r=()=>{const p=new URLSearchParams;return o?(p.append("isPro","true"),t.referer&&p.append("referer",t.referer),t.fullURL&&p.append("fullUrl",t.fullURL)):(t.type&&p.append("type",t.type),t.layout&&p.append("layout",t.layout),t.seed_url&&p.append("seedUrl",t.seed_url)),p.append("locale",i),p.append("country",t.country),t.state&&p.append("state",t.state),`cm/${t.cmId}/modalconfig?${p.toString()}`};n(Jn.actions.fetchConfigStart());const l=window.setTimeout(()=>{T.error("[CCM] Dead CM timeout triggered - modal did not start loading within 5 seconds"),lt("Dead CM timeout - modal did not start loading within 5 seconds",n),n(Jn.actions.fetchConfigFailure("Dead CM timeout"))},5e3);n(Da(l));const c=new AbortController,d=window.setTimeout(()=>c.abort(),5e3);fetch(`${hn(t)}${r()}`,{signal:c.signal}).then(p=>{if(clearTimeout(d),!p.ok)throw new Error(`Failed to fetch configuration: HTTP ${p.status}`);return p.json()}).then(p=>o?p:a(p)).then(p=>Dl(p)).then(p=>{n(Ie());const u=window.setTimeout(()=>{T.error("[CCM] Partial load timeout triggered - modal did not fully load within 30 seconds"),lt("Partial load timeout - modal did not fully load within 30 seconds",n)},3e4);n(Fa(u)),n(Jn.actions.fetchConfigSuccess(p))}).catch(p=>{clearTimeout(d),n(Ie());const u=p.name==="AbortError"?"modalconfig API request timed out":`modalconfig API failed: ${p.message}`;T.error("[CCM] Error fetching config: {}",u),lt(u,n),n(Jn.actions.fetchConfigFailure(u))})};let Pe=!1;const Xe=({ariaLabel:n})=>{var u;const e=rn(),t=y(f=>f.configSlice.config),i=Ge(),o=y(f=>f.globalStateSlice.effectiveLocale),a=y(f=>f.globalStateSlice.currentView),r=y(f=>f.configSlice.loaded),l=y(f=>f.configSlice.error),c=R(null),d=It(i,t==null?void 0:t.config);if(W(()=>{var f;Pe&&(r?((f=c.current)==null||f.focus(),Pe=!1):l&&(Pe=!1))},[r,l]),!((u=t.config.languages)!=null&&u.length))return null;const p=f=>{const x=f.target.value;Pe=!0,e(oo(x)),Ro(e,{rootProps:i,effectiveLocale:x}),Be({messageType:Kn.CREATE_COOKIE,data:{name:"notice_language",value:x}})};if(i.cmType==="PRO"||i.cmType==="ADV"&&d){const f=t.contents.language_selector_label||t.contents.configuration_language_selector_label_key||"Select language";return s(E,{children:[s("div",{className:"languageSelectorContainer",children:[s("label",{htmlFor:"taLanguageListBox",children:f}),s("select",{ref:c,id:"taLanguageListBox",className:"languageSelectorAccessible p-semibold",value:o,onChange:p,children:t.config.languages.map(g=>s("option",{value:g.code,children:g.label},g.code))})]}),a!=="advancedPanel"&&a!=="sliderPanel"&&s("div",{"aria-hidden":"true",className:"gwt-HTML divider",children:s("hr",{})})]})}return s("div",{className:"center",children:s("select",{ref:c,className:"language-selector","aria-label":n||"Language List Box",value:o,onChange:p,children:t.config.languages.map(f=>s("option",{value:f.code,children:f.label},f.code))})})};function Fl(){var f;const n=rn(),e=y(g=>g.configSlice.config),t=y(g=>g.globalStateSlice.advRootProps),i=y(g=>g.globalStateSlice.startingView),o=R(null),a=R(null),r=j(t);W(()=>{o.current&&(t!=null&&t.action)&&Ye(o.current,t)},[t==null?void 0:t.action]);const l=async({isAccept:g})=>{const x=t!=null&&t.is_gpc||t!=null&&t.is_dnt?(t==null?void 0:t.cpra_consent)??"":"";let b=!1,h={},m={};if(g){const _=go(e,t);({activeCategories:h,activeVendors:m}=Ce({level:_,allCategories:e.categories,dispatch:n}))}i==="loyaltyMessage"?(b=!0,En()):fl({activeCategoriesUpdated:h,activeVendorsUpdated:m}),Be({messageType:g?Kn.FINPROG_YES:Kn.FINPROG_NO,data:bl(x,b,t.cpra_source??"")})},c=()=>l({isAccept:!1}),d=()=>l({isAccept:!0}),p={key:"no",label:e.contents.loyaltymessage_opt_out_financial_incentive_text_key||So,onClick:c,styleName:`loyaltyMessageNo${r?"Accessible ta-modal-primary-btn accessibleButton":""}`},u={key:"yes",label:e.contents.loyaltymessage_opt_in_financial_incentive_text_key||ko,onClick:d,styleName:`loyaltyMessageYes${r?"Accessible ta-modal-primary-btn accessibleButton":""}`};return s(E,{children:s("div",{class:`loyaltyMessage${r?" loyaltyMessageAccessible":""}`,children:[s("div",{class:"loyaltyMessageHeader",children:s("div",{ref:o,children:s(U,{containerRef:o,html:e.contents.loyaltymessage_title_key})})}),r&&V(e.contents.configuration_has_language_selector_key)&&((f=e.config.languages)==null?void 0:f.length)>0&&s(Xe,{}),s("div",{class:"loyaltyMessageBody",children:s("div",{ref:a,children:s(U,{containerRef:a,html:e.contents.loyaltymessage_contents_key})})}),s(qe,{buttons:[p,u],className:"loyaltyMessageBottom"})]})},"finprog")}function Nl(){const n=rn(),{createEventHandlers:e}=We(),t=y(g=>g.configSlice.config),i=y(g=>g.globalStateSlice.advRootProps),o=R(null),[a,r]=_n({}),l=Qe(),c=Ke(),d=j(i);W(()=>{p()},[]);function p(){const g=e({"#privacyPolicy":{click:()=>{n(un("privacyPolicyPanel"))}},"#cookieSettingsLink":{click:l},"#advancedSettingsLink":{click:c}});r(g)}const u=()=>{En()},f={key:"close",id:"gwt-debug-close_id",label:d?`<span class="tc-icon tc-icon-cancel-close" aria-hidden="true"></span><span class="text-white pSemiBold">${t.contents.optoutdonemessage_closebutton_label_key||wi}</span>`:t.contents.optoutdonemessage_closebutton_label_key||wi,onClick:u,styleName:d?"ta-modal-primary-btn accessibleButtonPrimary":"close"};return s(E,{children:[s("div",{class:d?"accessibleContentfield":"contentfield",ref:o,children:t.contents.optoutdonemessage_contents_key&&s(U,{containerRef:o,html:t.contents.optoutdonemessage_contents_key,eventHandlers:a})}),s(qe,{buttons:[f]})]},"optout-done-message")}function Il(){const n=y(t=>t.configSlice.config),e=R(null);return s(E,{children:[s("div",{ref:e,children:s(U,{containerRef:e,html:n.contents.privacypolicypanel_content_key})}),s("br",{}),s(Se,{})]})}function Rl(){const n=y(w=>w.configSlice.config),e=rn(),{heightList:t,levelHeightCheckList:i,level:o,sliderPanelHeight:a,sliderWrapperHeight:r,sliderTop:l}=y(w=>w.preferencesSlice.sliderState),c=y(w=>w.globalStateSlice.isMobile),d=y(w=>w.globalStateSlice.advRootProps),p=j(d),u=R(null),f=R(null),g=R(null),x=R(null);W(()=>{D(o)},[r,o]),W(()=>(document.addEventListener("dragover",on),window.addEventListener("resize",Z),Z(),()=>{document.removeEventListener("dragover",on),window.removeEventListener("resize",Z)}),[]);const b=n.categories??[],h=$(()=>b.map(()=>pe()),[b]),m=$(()=>b.map(()=>pe()),[b]),_=Fo(),k=$(()=>b.map(w=>_(w.description)),[b,_]),A=()=>p?`sliderBody${c?" radioButtonBody":""}`:"prefPanel",S=()=>p?`sliderLeft${c?" radioButtonPanel":""}`:"left",M=()=>p?"sliderWrapperAccessible":"sliderWrapper",z=()=>p?`sliderOptions${c?" checkOptions":""}`:`options ${c?"checkOptions":""}`,O=()=>p?"sliderRight":"right",F=()=>c&&!p,N=()=>p&&c,B=w=>{w>=0&&w<=b.length-1&&w!==o&&Ce({level:w,allCategories:n.categories,dispatch:e})},C=(w,P)=>{e(Xr({[w]:P}))},D=w=>{const P=i[w];G(P)},G=w=>{C("sliderTop",w),C("sliderPanelHeight",w)},J=w=>{let P=0,K=1/0,Q=0;for(let an=0;an<t.length;an++){const cn=Q,Dn=Q+t[an];if(w>=cn&&w<(cn+Dn)/2){const Fn=Math.abs(cn-w);Fn<K&&(K=Fn,P=an)}else if(w>=(cn+Dn)/2&&w<Dn){const Fn=Math.abs(Dn-w);Fn<K&&(K=Fn,P=an+1)}Q+=t[an]}return P},Z=()=>{requestAnimationFrame(()=>{if(x.current&&u.current){const w=[...x.current.querySelectorAll("div")].map(Q=>{let an=0;const cn=Q.getElementsByClassName("truncatedCat")[0];return cn&&(an=parseFloat(window.getComputedStyle(cn).marginBottom)),Q.offsetHeight+an});C("heightList",w);const P=[];let K=0;for(let Q=0;Q<b.length;Q++)P.push(K),K+=w[Q];C("levelHeightCheckList",P),C("sliderWrapperHeight",P[P.length-1]+u.current.offsetHeight/2)}})},L=w=>{if(g.current===null)return;const K=g.current.getBoundingClientRect().top,Q=w.clientY-K,an=J(Q);B(an),D(an)},on=w=>{if(g.current===null||u.current===null)return;w.dataTransfer.dropEffect="move";const P=g.current.getBoundingClientRect().top,K=(u.current.clientHeight||0)/2;let Q=w.clientY-P-K;Q=Math.max(0,Q),Q=Math.min((g.current.clientHeight??0)-K,Q),G(Q)},Ln=On(),Hn=w=>{document.body.classList.add("dragging"),setTimeout(()=>{var P;return(P=u.current)==null?void 0:P.classList.remove("draggable")},5),w.dataTransfer.setDragImage(Ln,0,0)},jn=()=>{if(g.current===null||u.current===null)return;const w=g.current.getBoundingClientRect().top,P=u.current.getBoundingClientRect().top-w,K=J(P);B(K),D(K),u.current.classList.add("draggable"),document.body.classList.remove("dragging")};function On(){const w=document.createElement("img");return w.style.opacity="0",w.setAttribute("src","data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"),w}const Sn={},I=n.contents.preftable_slider_tooltip_label_key;I&&(n.contents.css_key.includes(".titleHolder")?Sn["data-title"]=I:Sn.title=I);const gn=w=>{if(c){if(w===o)return"selected";if(w>o)return"disabled"}return w<=o?"active":""},wn=w=>{const K=w.currentTarget.dataset.index;n.contents.preftable_only_radio_button_clickable_key&&w.target.closest(`#catDetails${K}`)||K&&B(parseInt(K))},kn=()=>{B(o-1)},zn=()=>{B(o+1)};return s(E,{children:[s("div",{className:A(),children:[s("div",{className:S(),children:[s("div",{ref:g,onClick:L,className:M(),style:{height:`${r}px`,...c&&{display:"none"}},children:s("div",{className:"ui-slider ui-slider-vertical ui-widget ui-widget-content ui-corner-all",style:{height:`${r}px`},children:[s("div",{ref:f,className:"ui-slider-range ui-widget-header ui-slider-range-max",style:{height:`${a}px`}}),s("div",{style:{height:`${r}px`,width:"19px",left:"-8px",position:"absolute"},children:s("div",{children:[s("input",{type:"text",tabIndex:-1,"aria-hidden":"true",style:{opacity:0,height:"1px",width:"1px",zIndex:-1,overflow:"hidden",position:"absolute"}}),typeof o=="number"&&s("div",{ref:u,onDragStart:Hn,onDragEnd:jn,tabindex:0,role:"slider","aria-valuetext":b.length>0?`${b[o].displayText}. ${b[o].description}`:void 0,"aria-valuenow":o,"aria-valuemax":b.length-1,"aria-valuemin":0,"aria-label":"Cookie Usage","aria-controls":"rightPanel","aria-orientation":"vertical",className:"ui-slider-handle ui-state-default ui-corner-all sliderImage titleHolder draggable",draggable:!0,...Sn,style:{position:"absolute",top:`${l}px`},...kl(kn,zn),children:p?s("svg",{width:"21",height:"20",viewBox:"0 0 21 20",fill:"none",xmlns:"http://www.w3.org/2000/svg",className:"slider-icon",id:"sliderHandleImage",role:"presentation",children:[s("rect",{x:"19.5",y:"1",width:"18.002",height:"18",rx:"9",transform:"rotate(90 19.5 1)",fill:"white"}),s("rect",{x:"19.5",y:"1",width:"18.002",height:"18",rx:"9",transform:"rotate(90 19.5 1)",stroke:"currentColor",strokeWidth:"2"})]}):V(n.contents.configuration_is_accessible_key)&&s("img",{src:`${hn()}asset/handle.jpg`,className:"gwt-Image",alt:"Cookie Preference button",role:"button","aria-label":"Cookie Preference button"})})]})})]})}),s("div",{ref:x,className:z(),...N()&&{role:"radiogroup"},children:b.map((w,P)=>s("div",{"data-index":P,className:gn(P),...F()&&{role:"radio",tabIndex:0,style:{cursor:"pointer"},onClick:wn,onTouchStart:wn},children:s("span",{id:"catDetails"+P,style:"margin-bottom:0px; cursor: default;",className:N()?"accessibleCategoryHeadingName":"",children:[p?s("h3",{ref:h[P],className:"sliderCatTitle",children:s(U,{containerRef:h[P],html:w.displayText})},P):s("h2",{ref:h[P],children:s(U,{containerRef:h[P],html:w.displayText})},P),s("p",{id:"sliderCatDef_"+(P+1),className:N()?"sliderCatDef":"truncatedCat",ref:m[P],children:s(U,{containerRef:m[P],html:k[P]})}),N()&&s("div",{className:"radioAlignRight",id:"radioPlaceholder"+P,children:s("input",{type:"radio",name:"categoryGroup",value:P<=o?"on":"off",className:"categoryChoice",id:"categoryRadio"+P,"aria-checked":P===o?"true":"false","aria-posinset":P+1,"aria-setsize":b.length,tabIndex:0,onClick:()=>B(P),onTouchStart:()=>B(P)})})]})},P))}),s("br",{"aria-hidden":"true",className:"br-clear"})]}),s("div",{className:O(),id:"rightPanel",children:[s("h3",{tabindex:p?void 0:0,children:s("span",{className:p?"sliderAllowed":"allowedTitle",children:n.contents.preftable_allowed_header_key})}),s("div",{children:s("ul",{className:p?"sliderList":void 0,tabindex:p?void 0:0,children:b.slice(0,o+1).map(w=>w.details.map((P,K)=>s("li",{className:p?"sliderAllowedItem":void 0,children:P.text},K)))})}),s("br",{"aria-hidden":"true",className:"spacing first-br"}),(()=>{const w=o+1>=b.length;return s(E,{children:[s("h3",{className:`not ${w?"displayNone":""}`,tabindex:p||w?void 0:0,"aria-hidden":w,children:s("span",{className:p?"sliderNotAllowed":"notAllowedTitle",style:`visibility: ${w?"hidden":"visible"};`,children:n.contents.preftable_notallowed_header_key})}),s("div",{className:`${w?"displayNone":""}`,children:s("ul",{className:p?"sliderList":void 0,tabindex:p||w?void 0:0,"aria-hidden":w,children:!w&&b.slice(o+1,b.length).map(P=>P.details.map((K,Q)=>s("li",{className:p?"sliderNotAllowedItem":void 0,children:K.text},Q)))})})]})})(),!p&&s("br",{"aria-hidden":"true",className:"spacing br-clear"})]}),!p&&s("br",{"aria-hidden":"true",className:"spacing br-clear"})]}),!p&&s("br",{"aria-hidden":"true",className:"spacing br-clear"}),s(Se,{})]},"slider")}function Ml(){const n=y(p=>p.globalStateSlice.startingView),e=y(p=>p.globalStateSlice.previousView),t=y(p=>p.preferencesSlice.savedCategoriesConsent),i=y(p=>p.preferencesSlice.savedVendorsConsent),o=rn(),a=qt(),r=n==="twoStepPanel"||n==="proTwoStepPanel";return{handleBack:()=>{!r&&e?(o(Jr({savedCategoriesConsent:t,savedVendorsConsent:i})),st(!1),o(un(e))):En()},handleNo:()=>{st(!1),Be({messageType:Kn.TWO_STEP_NO,data:""}),En()},handleYes:()=>{r?(Be({messageType:Kn.TWO_STEP_YES,data:""}),En()):(a.logInformationalActivity(Me.TWO_STEP_OPT_IN),st(!0))}}}function Bl(){var f;const n=y(g=>g.configSlice.config),e=y(g=>g.globalStateSlice.advRootProps),t=R(null),i=R(null),{handleBack:o,handleNo:a,handleYes:r}=Ml(),l=j(e),c=Rn(n.contents.twosteppanel_body_key);W(()=>{t.current&&(e!=null&&e.action)&&Ye(t.current,e)},[e==null?void 0:e.action]);const d={key:"back",label:n.contents.twosteppanel_back_key||Gs,onClick:o,styleName:`twoStepBack${l?" accessibleButton":""}`},p={key:"no",label:n.contents.twosteppanel_no_key||So,onClick:a,styleName:`twoStepNo${l?" accessibleButton":""}`},u={key:"yes",label:n.contents.twosteppanel_yes_key||ko,onClick:r,styleName:`twoStepYes${l?" accessibleButton":""}`};return s(E,{children:s("div",{class:`twoStep${l?" accessible":""}`,children:[s("div",{class:"twoStepHeader",children:s("div",{ref:t,children:s(U,{containerRef:t,html:n.contents.twosteppanel_header_key})})}),l&&V(n.contents.configuration_has_language_selector_key)&&((f=n.config.languages)==null?void 0:f.length)>0&&s(Xe,{}),s("div",{class:"twoStepBody",ref:i,children:s(U,{containerRef:i,html:c})}),s(qe,{buttons:[d,p,u],className:`twoStepBottom${l?" accessibleTwoStepButtonContainer":""}`})]})},"two-step")}function Hl({subsection:n,toggleState:e,onToggle:t}){var u,f,g;const i=y(x=>x.configSlice.config),o=or(n.name),a=sr(n),r=((u=i.contents)==null?void 0:u.preftable_in_label_key)||"Yes",l=((f=i.contents)==null?void 0:f.preftable_out_label_key)||"No",c=x=>{t(n.name,x)},d=`${l} ${o}`,p=`${r} ${o}`;return s("div",{className:"gpp_subsection",children:[s("div",{className:"gwt-HTML",children:o}),s(Ze,{toggleState:e,onLabel:r,offLabel:l,onToggle:c,ariaLabelOn:p,ariaLabelOff:d,isVisible:!0,removeSwitchParentRole:!1,removeSwitchParentAriaLabel:!1,isAccessibleSwitch:!1,isVendor:!1,isGppPurposeSwitch:!0,switchAriaLabel:`${o} ${(((g=i.contents)==null?void 0:g.preftable_choice_label_key)||"Choice").toLowerCase()}`})]},a)}function zl({purpose:n,vendors:e,isExpanded:t,toggleState:i,subsectionStates:o,onToggleExpand:a,onToggleActive:r,onToggleSubsection:l}){var z,O;const c=y(F=>F.globalStateSlice.advRootProps),d=y(F=>F.configSlice.config),p=R(null),u=j(c),f=hi(n.name),g=Rn(n.description),x=((z=d.contents)==null?void 0:z.preftable_in_label_key)||"Yes",b=((O=d.contents)==null?void 0:O.preftable_out_label_key)||"No",h=()=>s("div",{className:"accessibleCategoryHeadingName",children:[s("div",{className:"gwt-HTML",children:s("h2",{ref:p,children:s(U,{containerRef:p,html:f})})}),s("div",{className:"gwt-HTML",children:s("p",{dangerouslySetInnerHTML:{__html:g}})})]}),m=()=>{var F;return u?s(Ze,{toggleState:i,onLabel:x,offLabel:b,onToggle:r,ariaLabelOn:`${x} ${f}`,ariaLabelOff:`${b} ${f}`,isVisible:!0,removeSwitchParentRole:!1,removeSwitchParentAriaLabel:!1,isAccessibleSwitch:!1,isVendor:!1,isGppPurposeSwitch:!0,switchAriaLabel:`${f} ${(((F=d.contents)==null?void 0:F.preftable_choice_label_key)||"Choice").toLowerCase()}`}):null},_=()=>{const F=d.contents.preftable_view_detailed_settings_key||"View detailed settings",N=t?`Hide ${f}`:`View ${f}`;return s("div",{className:"viewcookie",children:[s("button",{type:"button",className:`ta-modal-accordion header${t?" open":""}`,onClick:a,"aria-expanded":t,"aria-label":N,...qn(a),children:[s("img",{src:`${hn()}asset/chevron-right-grey.svg`,className:`normalArrow${t?" activeArrow":""}`,alt:""}),s("div",{className:"gwt-HTML",style:{visibility:"visible"},children:F})]}),s("div",{className:"accordionContentContainer table",style:t?{}:{display:"none"},"aria-hidden":!t,children:[mi(n)&&k(),e.length>0&&s("div",{children:s("div",{className:"left",children:s("div",{className:"gwt-HTML",children:s("p",{children:n.detailed_desc})})})}),e.length>0?A():S()]})]})},k=()=>{const F=ar(n);return F.length===0?null:s(E,{children:[s("div",{className:"gwt-HTML subsections_header",style:"",children:f}),s("div",{className:"subsections",style:"",children:F.map(N=>{const B=o[N.name]??null;return s(Hl,{subsection:N,toggleState:B,onToggle:(C,D)=>l(C,D)},`${n.id}-${N.name}`)})})]})},A=()=>e.length===0?null:s("table",{className:"prefTable optoutless",children:[s("colgroup",{children:[s("col",{width:"2%"}),s("col",{width:"28%"}),s("col",{width:"70%"})]}),s("thead",{children:s("tr",{children:[s("th",{scope:"col",children:" "}),s("th",{scope:"col",children:"Company"}),s("th",{scope:"col",children:"Domain"})]})}),s("tbody",{children:e.map((F,N)=>s(No,{vendor:F,index:N,categoryName:hi(n.name),categoryId:`gpp-${n.id}`,toggleState:null,isRequired:!1,isGppContext:!0},F.id))})]}),S=()=>{var F;return s("div",{className:"p-semibold noCookiesMessage",children:s("p",{children:((F=d.contents)==null?void 0:F.preftable_cookies_dont_exist_label_key)||To})})},M=mi(n)||e.length>0;return s("div",{className:"gpp_category",children:[s("div",{children:[h(),m()]}),M&&_()]})}function Fe(){const n=R(null),e=y(o=>o.globalStateSlice.advRootProps),t=$t("popupMsg"),i=t&&ve("isReconsentEvent");return j(e)?s("div",{class:"ta-feedbackMessageContainer",children:s("div",{class:"tc-repop-msg","aria-hidden":!i,style:i?"":"display: none;",children:i&&s("div",{class:"gwt-HTML ta-feedback ta-feedback-informational",children:[s("i",{class:"tc-warning"}),t]})})}):i?s("div",{class:"repopMessage",children:s("div",{class:"gwt-HTML",ref:n,children:s(U,{containerRef:n,html:t})})}):null}function $l(){var k,A;const n=rn(),e=y(S=>S.globalStateSlice.advRootProps),t=y(S=>S.configSlice.config),i=y(S=>S.preferencesSlice.expandedGppPurposes),o=y(S=>S.preferencesSlice.activeGppPurposes),a=y(S=>S.preferencesSlice.activeGppSubsections),r=R(null),l=R(null),c=j(e),d=Rn(((k=t.contents)==null?void 0:k.preftable_cookie_details_message_key)||'Please choose whether this site may use cookies or related technologies such as web beacons, pixel tags, and Flash objects ("Cookies") as described below. You can learn more about how this site uses cookies and related technologies by reading our privacy policy linked below.'),{gppPurposes:p,gppCountryState:u}=$(()=>{const S=t.config.gppCountryStates||[],M=S.length>0?S[0]:void 0;return{gppPurposes:(M==null?void 0:M.purposes)||[],gppCountryState:M}},[t.config.gppCountryStates]),f=$(()=>new Set(Object.keys(i).filter(S=>i[S])),[i]),g=$(()=>{const S={};return Object.entries(a).forEach(([M,z])=>{const O=M.split("_");if(O.length>=2){const F=O.slice(0,-1).join("_"),N=O[O.length-1];S[F]||(S[F]={}),S[F][N]=z}}),S},[a]),x=$(()=>{if(!u)return{};const S={};return p.forEach(M=>{if(M.value>=0){const z=t.categories.find(O=>O.priority===M.value);S[M.id]=(z==null?void 0:z.vendors)||[]}else S[M.id]=[]}),S},[p,u,t.categories]),b=$(()=>[...p].filter(S=>S.value!==Re.NOT_APPLICABLE).sort((S,M)=>S.order-M.order),[p]),h=ln(S=>{n(ts(S))},[n]),m=ln((S,M)=>{n(ns({purposeId:S,enabled:M}))},[n]),_=ln((S,M,z)=>{n(es({purposeId:S,subsectionName:M,enabled:z}))},[n]);return W(()=>{p.length>0&&n(is(p))},[p,n]),W(()=>{Ye(r.current,e)},[e]),s(E,{children:s("div",{className:`mainContent gppMainContent${c?" gpp_accessibility":""}`,role:"main","aria-labelledby":"cookieMainHeader",children:[s("div",{ref:r,className:"modal_header",children:[s("div",{children:s("div",{className:"mainHeader consentHeader accessibleHeader",id:"cookieMainHeader",children:((A=t.contents)==null?void 0:A.preftable_cookies_and_related_technologies_message_key)||"Cookies and Related Technologies on This Site"})}),s("button",{type:"button",className:"gwt-Button ta-modal-tertiary-btn",id:"trustarc-internal-close-button","aria-label":"Close button",style:"visibility: hidden"})]}),s("div",{className:"modal_body",children:[s("img",{className:"logo","aria-label":"Logo",src:`${hn()}asset/trustarc-logo-big.svg`}),s("div",{className:"prefPanel gsPrefPanelNoLogo gppPrefPanel prefTableAdvanced",children:[s("div",{className:"layoutLabel",children:s("div",{className:"gwt-HTML",ref:l,children:s(U,{containerRef:l,html:d})})}),s(fe,{}),s(Fe,{}),ve("gpp.gppShowCategories")&&s(Io,{}),b.map(S=>{const M=rr(S),z=f.has(S.id),O=o[S.name]??null,F=g[S.name]||{},N=x[S.id]||[];return s(zl,{purpose:S,vendors:N,isExpanded:z,toggleState:O,subsectionStates:F,onToggleExpand:()=>h(S.id),onToggleActive:B=>m(S.name,B),onToggleSubsection:(B,C)=>_(S.name,B,C)},M)})]})]}),s(Se,{})]})},"gpp-panel")}const ct={"":E,euDisclosurePanel:Tl,sliderPanel:Rl,advancedPanel:Io,gppPanel:$l,optoutDoneMessage:Nl,twoStepPanel:Bl,privacyPolicyPanel:Il,loyaltyMessage:Fl};function Vl(){const n=y(a=>a.configSlice.config),e=y(a=>a.globalStateSlice.currentView),t=R(null),i=R(null);return e==="advancedPanel"||e==="iabManageSettingsPanel"?s(E,{children:s("div",{className:"gwt-PopupPanel",ref:t,style:{position:"absolute",top:"50%",left:"50%",transform:"translate(-50%, -50%)",overflow:"visible",visibility:"visible",minWidth:"200px",maxWidth:"min(600px, 80vw)",width:"max-content"},children:s("div",{class:"popupContent",children:s("div",{class:"gwt-HTML reset-html",ref:i,children:s(U,{containerRef:i,html:n.contents.preftable_disabled_submit_message_key||Ao})})})})},"disabled-submit-message"):null}const Ul={euDisclosurePanel:"disclosure_title_key",sliderPanel:"preftable_title_key",advancedPanel:"preftable_title_key",optoutDoneMessage:"optoutdonemessage_title_key",privacyPolicyPanel:"privacypolicypanel_title_key"};function pt(){var B;const n=rn(),{createEventHandlers:e}=We(),t=y(C=>C.configSlice.config),i=y(C=>C.globalStateSlice.currentView),o=Ul[i]??"",a=y(C=>C.globalStateSlice.advRootProps),r=Oo(),[l,c]=_n(""),[d,p]=_n({}),[u,f]=_n(""),g=R(null),x=R(null),b=y(C=>C.uiStateSlice.titleLogoExist),h=Zn(),m=Qe(),_=Ke();W(()=>{A(),S(t),M(t)},[t,o]),W(()=>{if(!h||!l)return;const C=x.current||g.current;C&&C.classList.contains("gsHeader")&&z(C)},[h,l]),W(()=>{if(l){j(a)&&k();const C=x.current||g.current;if(!C)return;Ye(C,a)}},[l,a]);function k(){const C=x.current||g.current;if(!C)return;const D=V(t.contents.preftable_show_accept_all_button_upper_key),G=V(t.contents.preftable_show_decline_all_button_upper_key),J=D&&G,Z=C.querySelector("#accept_all_button");if(Z&&D){const on=He(t.contents.preftable_accept_all_button_lower_text_key||vo);Z.innerHTML=on,Z.id="accept_all_button_accessible",Z.className="ta-modal-primary-btn accessibleButtonPrimary",Z.setAttribute("tabindex","0"),Z.setAttribute("role","button"),J&&(Z.style.display="inline-flex")}const L=C.querySelector("#decline_all_button");if(L&&G){const on=He(t.contents.preftable_decline_all_button_lower_text_key||Co);L.innerHTML=on,L.id="decline_all_button_accessible",L.className="ta-modal-primary-btn accessibleButtonPrimary",L.setAttribute("tabindex","0"),L.setAttribute("role","button"),J&&(L.style.display="inline-flex")}}function A(){if(t.contents&&o!==""){const C=t.contents[o]||"";c(C)}}function S(C){const D=e({"#accept_all_button,#accept_all_button_accessible":{click:()=>{Ct({currentView:i,cmConfig:C,rootProps:a,consent:r,dispatch:n})}},"#decline_all_button,#decline_all_button_accessible":{click:async()=>{const G=Ce({level:0,allCategories:C.categories,dispatch:n});await(r==null?void 0:r.doConsent(yn.DECLINE_ALL,G))}},"#cookieSettingsLink":{click:m},"#advancedSettingsLink":{click:_}});p(D)}function M(C){const D=j(a),G=V(C.contents.preftable_show_accept_all_button_upper_key),J=V(C.contents.preftable_show_decline_all_button_upper_key);f(`
      #${D&&G?"accept_all_button_accessible":"accept_all_button"} {
        display: ${G?D?"inline-flex":"inline":"none"} !important;
      }
      #${D&&J?"decline_all_button_accessible":"decline_all_button"} {
        display: ${J?D?"inline-flex":"inline":"none"} !important;
      }
    `)}function z(C){n(Va());const D=[...C.children];for(const L of D)L.className.toLowerCase()==="logo"?G(L):L.tagName.toLowerCase()==="p"&&J(L);Z();function G(L){const on=document.createElement("div");for(on.classList.add("logo"),on.classList.add("gsLogo");L.firstChild;)on.appendChild(L.firstChild);L.replaceWith(on),n($a(!0))}function J(L){Bn(a)&&n(za({id:L.id,html:L.outerHTML})),L.remove()}function Z(){if(!V(t.contents.preftable_has_accept_all_button_key))return;const L=document.createElement("a");L.className="acceptallbutton",L.tabIndex=0,L.textContent=t.contents.preftable_accept_all_button_text_key??ll,L.onclick=()=>Ct({currentView:i,cmConfig:t,rootProps:a,consent:r,dispatch:n}),C.appendChild(L)}}function O(C){let D="mainHeader consentHeader";if(j(a)){D+=" accessibleHeader";const G=V(t.contents.preftable_show_accept_all_button_upper_key),J=V(t.contents.preftable_show_decline_all_button_upper_key),Z=(G?1:0)+(J?1:0);Z===2?D+=" twoButtonPresent":Z===1&&(D+=" oneButtonPresent")}return h&&(C==="primary"&&Pn(a)&&i==="advancedPanel"?D+=" hiddenElement hidden":(b||(D+=" gsHeaderNoLogo"),D+=" gsHeader")),D}const F=!Nt(i),N=Rn(l);return F?s(E,{children:[s("div",{className:O("primary"),ref:g,id:Do(i,Pn(a)),children:s(U,{containerRef:g,html:N,css:u,eventHandlers:d})},"header"),j(a)&&V(t.contents.configuration_has_language_selector_key)&&((B=t.config.languages)==null?void 0:B.length)>0&&s(Xe,{}),h&&Pn(a)&&i==="advancedPanel"&&s("div",{className:O("secondary"),ref:x,children:s(U,{containerRef:x,html:N,css:u,eventHandlers:d})},"header-duplicate")]}):null}function Gl(){var k;const n=rn(),e=y(A=>A.globalStateSlice.advRootProps),t=y(A=>A.configSlice.config),{createEventHandlers:i}=We(),[o,a]=_n({}),r=Zn(),l=Qe(),c=Ke();W(()=>{var A;(A=t.contents).privacypolicypanel_show_privacy_policy_footer_link_key??(A.privacypolicypanel_show_privacy_policy_footer_link_key="true"),f()},[t]);const d=R(null),p=R(null),u=R(null);function f(){const A=i({"#privacyPolicy":{click:()=>{n(un("privacyPolicyPanel"))}},"#cookieSettingsLink":{click:l},"#advancedSettingsLink":{click:c}});a(A)}function g(){let A=j(e)?"footerAccessibleContainer":"footer";return r&&(A+=" gsFooter"),A}function x(){return j(e)?"footerAccessible":""}const b=Rn(t.contents.footerpanel_left_key),h=Rn(t.contents.footerpanel_right_key),m=Rn(t.contents.footerpanel_privacy_policy_key),_=s(E,{children:[s("div",{class:j(e)?"footerLeft":"left",ref:d,children:s(U,{containerRef:d,html:b,eventHandlers:o})}),V(t.contents.footerpanel_has_language_selector_key)&&((k=t.config.languages)==null?void 0:k.length)>0&&s(Xe,{ariaLabel:t.contents.footerpanel_language_selector_aria_label_key}),j(e)&&s("div",{class:"footerDivider"}),s("div",{class:j(e)?"footerRight":"right",ref:p,children:s(U,{containerRef:p,html:h,eventHandlers:o})}),V(t.contents.privacypolicypanel_show_privacy_policy_footer_link_key)&&t.contents.footerpanel_privacy_policy_key&&s("div",{class:"privacypolicy",ref:u,children:s(U,{containerRef:u,html:m,eventHandlers:o})}),s("br",{class:"spacing"}),s("div",{"aria-hidden":"true",style:"display: none;"})]});return s(E,{children:s("div",{class:g(),style:"",children:j(e)?s("div",{class:x(),role:"contentinfo",children:_}):_})},"footer")}const dt=({routes:n})=>{const e=y(i=>i.globalStateSlice.currentView),t=n[e];return T.info("currentView: {}",e),s(t,{},e)},jl=()=>s("svg",{width:"32",height:"32",viewBox:"0 0 32 32",fill:"none",xmlns:"http://www.w3.org/2000/svg","aria-hidden":"true",children:s("path",{d:"M16.0015 28C14.0436 27.9983 12.1158 27.5176 10.3863 26.5998C8.65688 25.6819 7.17828 24.3549 6.07951 22.7343C4.98074 21.1138 4.29518 19.249 4.08264 17.3027C3.87009 15.3563 4.13702 13.3875 4.86014 11.5681C5.58325 9.74857 6.74059 8.13365 8.23118 6.86418C9.72178 5.59471 11.5004 4.70925 13.4118 4.28504C15.3232 3.86083 17.3093 3.91077 19.197 4.43048C21.0847 4.95019 22.8165 5.9239 24.2415 7.26666C24.4287 7.45416 24.5339 7.70833 24.5339 7.97333C24.5339 8.23833 24.4287 8.4925 24.2415 8.68C24.1495 8.77434 24.0397 8.84932 23.9183 8.90052C23.7969 8.95171 23.6665 8.97809 23.5348 8.97809C23.4031 8.97809 23.2727 8.95171 23.1513 8.90052C23.03 8.84932 22.9201 8.77434 22.8281 8.68C21.1469 7.10754 18.9753 6.16236 16.6787 6.0036C14.3822 5.84483 12.1012 6.48217 10.2196 7.80832C8.33795 9.13447 6.97073 11.0684 6.34807 13.2846C5.72541 15.5008 5.88535 17.8638 6.80096 19.9759C7.71657 22.0879 9.33193 23.82 11.3751 24.8805C13.4182 25.941 15.7644 26.2652 18.0185 25.7984C20.2727 25.3316 22.2972 24.1024 23.7512 22.3178C25.2052 20.5331 25.9999 18.302 26.0015 16C26.0015 15.7348 26.1068 15.4804 26.2944 15.2929C26.4819 15.1054 26.7363 15 27.0015 15C27.2667 15 27.521 15.1054 27.7086 15.2929C27.8961 15.4804 28.0015 15.7348 28.0015 16C28.0015 19.1826 26.7372 22.2348 24.4868 24.4853C22.2363 26.7357 19.1841 28 16.0015 28Z",fill:"#E11A77",children:s("animateTransform",{attributeName:"transform",type:"rotate",from:"0 16 16",to:"360 16 16",dur:"1s",repeatCount:"indefinite"})})});function Mo({isPro:n=!1}={}){return n?s("div",{id:"loading2",role:"status","aria-label":"Loading ...",style:"display: flex; align-items: center; justify-content: center; min-height: 200px; height: 100%; width: 100%; background: #fff;",children:s(jl,{})}):s("div",{id:"loading2",style:"height: 100%; background: #fff; top: 0px; width: 100%;",children:s("img",{style:"position: absolute; top: 50%; left: 50%; margin-left: -15.5px; margin-top: -60px; ",id:"loadingImg",src:`${hn()}asset/loading.gif`,alt:"Loading ..."})})}function Wl(n=!0){const e=Ge(),t=y(o=>o.configSlice.config.config),i=It(e,t);W(()=>{var u;if(!n||!i)return;const o=(u=document.querySelector("div.trustarc_newcm_container"))==null?void 0:u.shadowRoot;if(!o){console.warn("Focus trap: Could not find shadow root");return}const a=o,r='button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',l=()=>[...a.querySelectorAll(r)].filter(Ps),c=()=>{const f=a.getRootNode();return f instanceof ShadowRoot?f.activeElement:document.activeElement},d=f=>{if(f.key!=="Tab")return;const g=l();if(g.length===0)return;const x=g[0],b=g[g.length-1],h=c();f.shiftKey?h===x&&(f.preventDefault(),b.focus()):h===b&&(f.preventDefault(),x.focus())},p=f=>{if(f.key==="Escape"){const g=a.querySelector("#trustarc-internal-close-button");g&&g.style.visibility!=="hidden"&&g.focus()}};return a.addEventListener("keydown",d),a.addEventListener("keydown",p),e.cmType!=="PRO"&&requestAnimationFrame(()=>{requestAnimationFrame(()=>{const f=l();f.length>0&&f[0].focus()})}),()=>{a.removeEventListener("keydown",d),a.removeEventListener("keydown",p)}},[n,i,e.cmType])}class Yl{constructor(e){Ae(this,"css");Ae(this,"position");Ae(this,"PROPERTY_REGEX",/^([a-zA-Z-]+)\s*:\s*(.+)$/);this.css=e,this.position=0}parseAllMediaQueries(){const e=[];for(this.position=0;this.position<this.css.length;){const t=this.findNextMediaQuery();t&&e.push(t)}return e}findNextMediaQuery(){const e=this.css.indexOf("@media",this.position);if(e===-1)return this.position=this.css.length,null;this.position=e;const t=this.findNextCharacter("{",this.position);if(t===-1)return this.position=e+6,null;const i=this.css.substring(e,t).trim(),o=this.findClosingBrace(t+1);if(o===-1)return this.position=t+1,null;const a=this.css.substring(t+1,o),r=this.parseMediaDeclaration(i,a);return r.startIndex=e,r.endIndex=o+1,this.position=o+1,r}updateQuoteState(e,t,i,o){return(e==='"'||e==="'")&&t!=="\\"&&(i?e===o&&(i=!1,o=""):(i=!0,o=e)),[i,o]}findNextCharacter(e,t){let i=t,o=!1,a="";for(;i<this.css.length;){const r=this.css[i],l=i>0?this.css[i-1]:"";if([o,a]=this.updateQuoteState(r,l,o,a),!o&&r===e)return i;i++}return-1}findClosingBrace(e){const t=[];let i=e;for(;i<this.css.length;){const o=this.css[i],a=i>0?this.css[i-1]:"";let r=!1,l="";if([r,l]=this.updateQuoteState(o,a,r,l),!r){if(o==="{")t.push(o);else if(o==="}"){if(t.length===0)return i;t.pop()}}i++}return-1}parseMediaDeclaration(e,t){const i=e.replace(/^@media\s+/,"").trim(),o=/^(only\s+)?(\w+)?/.exec(i),a=o&&o[2]||"all",r=this.parseConditions(i);return{type:a,conditions:r,cssRules:t,originalQuery:e,startIndex:-1,endIndex:-1}}parseConditions(e){const t=[];let i=0,o=0,a="",r=!1,l="";for(;i<e.length;){const c=e[i],d=i>0?e[i-1]:"";if([r,l]=this.updateQuoteState(c,d,r,l),r)o>0&&(a+=c);else if(c==="(")o===0?a="":a+=c,o++;else if(c===")")if(o--,o===0){if(a.trim()){const p=this.parseCondition(a.trim());p&&t.push(p)}a=""}else a+=c;else o>0&&(a+=c);i++}return t}parseCondition(e){const t=this.PROPERTY_REGEX.exec(e);if(!t)return T.warn("Invalid media condition:",e),null;const i=t[1],o=t[2].trim();return{property:i,value:o,originalCondition:`(${e})`}}}function Bo(n,e="data-injected"){return _o(n,{tag:"style",dataAttribute:e,insertWithPrecedence:!0})}const ki=["min-width","max-width","min-height","max-height"],ql=/@import\s+(?:"[^"]*"|'[^']*'|url\(\s*(?:"[^"]*"|'[^']*'|[^)]*)\s*\))[^;]*;?/g,Kl=/^@import\s+["']([^"']+)["']/,Ql=/^@import\s+url\(\s*["']([^"']+)["']\s*\)/,Zl=/^@import\s+url\(\s*([^)"'\s]+)\s*\)/;function Xl(n){var e,t,i;return((e=Kl.exec(n))==null?void 0:e[1])??((t=Ql.exec(n))==null?void 0:t[1])??((i=Zl.exec(n))==null?void 0:i[1])}const Jl=/url\(\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)'|([^)]+))\s*\)/g;function nc(n){let e="",t=0;for(;t<n.length;){const i=n.indexOf("@",t),o=n.indexOf("{",t);if(o===-1){e+=n.substring(t);break}if(i!==-1&&i<o){const a=i;let r;const l=n.indexOf(";",a),c=n.indexOf("{",a);l!==-1&&(c===-1||l<c)?r=l+1:c!==-1?(r=Ai(n,c),r===-1&&(r=n.length)):r=n.length,e+=n.substring(t,r),t=r}else{const a=ec(n,t,o),r=Ai(n,o);if(r===-1){e+=n.substring(t);break}const l=n.substring(t,a),c=n.substring(a,o).trim(),d=n.substring(o+1,r-1);e+=l,e+=tc(c,d),t=r}}return e}function ec(n,e,t){let i=t-1;for(;i>=e&&/\s/.test(n[i]);)i--;for(;i>=e;){if(n[i]==="}")return i+1;i--}return e}function Ai(n,e){let t=e+1,i=1,o=!1,a="";for(;t<n.length&&i>0;){const r=n[t],l=t>0?n[t-1]:"";(r==='"'||r==="'")&&l!=="\\"?o?r===a&&(o=!1,a=""):(o=!0,a=r):o||(r==="{"?i++:r==="}"&&i--),t++}return i===0?t:-1}function tc(n,e){const t=n.split(",").map(c=>c.trim()).filter(c=>c.length>0),i=c=>c==="body"||/^body(?![a-zA-Z0-9_-])/.test(c);if(!t.some(i))return`${n} {${e}}`;let a="";const r=[],l=[];for(const c of t)i(c)?r.push(c):l.push(c);l.length>0&&(a+=l.join(", ")+` {${e}}
`);for(const c of r){if(c==="body")a+=`body {}
`,a+=`.wrapper {${e}}`;else{const d=c.substring(4);a+=`.wrapper${d} {${e}}`}c!==r[r.length-1]&&(a+=`
`)}return a}function ic(n){const e=n.split("/"),t=[];for(const i of e)i===".."?t.pop():i!==""&&i!=="."&&t.push(i);return t.join("/")}function St(n,e){const t=e||hn();if(!t)return n;const i=t.endsWith("/")?t:`${t}/`,o=a=>a.replace(/\\.|"/g,r=>r==='"'?'\\"':r);return n.replace(Jl,(a,r,l,c)=>{const p=(r??l??c??"").trim(),u=/^(https?:)?\/\//.test(p),f=/^data:/.test(p),g=/^#/.test(p);if(u||f||g)return a;if(p.includes("../")){const x=ic(p);return`url("${o(i+x)}")`}if(p.startsWith("/")){let x="";try{x=new URL(i).origin}catch{const b=i.match(/^(https?:\/\/[^\/]+)/);if(b)x=b[1];else return`url("${o(i+p.substring(1))}")`}return`url("${o(x+p)}")`}return`url("${o(i+p)}")`})}function $n(n){const e=n.match(/(-?\d*\.?\d+)(?:px|%)/);return Math.round(parseFloat(e?e[1]:n))}function oc(n,e){const t=/([^{]+)(\{[^{}]*\})/g;return n.replace(t,(i,o,a)=>`${o.split(",").map(c=>c.trim()).map(c=>`${e} ${c}`).join(", ")}${a}`)}function ac(n){const e=new Set,t=new Set,i=new Set,o=new Set;for(const a of n){const r=a.conditions.find(p=>p.property==="min-width");if(r){const p=$n(r.value);e.add(p)}const l=a.conditions.find(p=>p.property==="max-width");if(l){const p=$n(l.value);t.add(p)}const c=a.conditions.find(p=>p.property==="min-height");if(c){const p=$n(c.value);i.add(p)}const d=a.conditions.find(p=>p.property==="max-height");if(d){const p=$n(d.value);o.add(p)}}return{minWidth:[...e].sort((a,r)=>a-r),maxWidth:[...t].sort((a,r)=>a-r),minHeight:[...i].sort((a,r)=>a-r),maxHeight:[...o].sort((a,r)=>a-r)}}function rc(n,e){const t=[];for(const o of e){if(!o.conditions.some(g=>ki.includes(g.property)))continue;let r="";const l=o.conditions.find(g=>g.property==="min-width");if(l){const g=$n(l.value);r+=`:where([data-breakpoint-min-width-${g}="true"])`}const c=o.conditions.find(g=>g.property==="max-width");if(c){const g=$n(c.value);r+=`:where([data-breakpoint-max-width-${g}="true"])`}const d=o.conditions.find(g=>g.property==="min-height");if(d){const g=$n(d.value);r+=`:where([data-breakpoint-min-height-${g}="true"])`}const p=o.conditions.find(g=>g.property==="max-height");if(p){const g=$n(p.value);r+=`:where([data-breakpoint-max-height-${g}="true"])`}if(r==="")continue;let u=oc(o.cssRules,r);const f=o.conditions.filter(g=>!ki.includes(g.property));if(f.length>0){const g=f.map(x=>x.originalCondition).join(" and ");!o.type||o.type==="all"?u=`@media ${g} {${u}}`:u=`@media ${o.type} and ${g} {${u}}`}t.push({startIndex:o.startIndex,endIndex:o.endIndex,replacement:u})}t.sort((o,a)=>a.startIndex-o.startIndex);let i=n;for(const{startIndex:o,endIndex:a,replacement:r}of t)i=i.slice(0,o)+r+i.slice(a);return i}function sc(n,e){const t=(a,r)=>{e.maxWidth.forEach(l=>{const c=`data-breakpoint-max-width-${l}`;a<=l?n.setAttribute(c,"true"):n.removeAttribute(c)}),e.minWidth.forEach(l=>{const c=`data-breakpoint-min-width-${l}`;a>=l?n.setAttribute(c,"true"):n.removeAttribute(c)}),e.maxHeight.forEach(l=>{const c=`data-breakpoint-max-height-${l}`;r<=l?n.setAttribute(c,"true"):n.removeAttribute(c)}),e.minHeight.forEach(l=>{const c=`data-breakpoint-min-height-${l}`;r>=l?n.setAttribute(c,"true"):n.removeAttribute(c)})},i=()=>{var r;const a=n.offsetHeight;if(a>0){const l=(r=self==null?void 0:self.document)==null?void 0:r.documentElement;l&&(l.style.setProperty("--truste-box-overlay-height",`${a}px`),l.style.setProperty("--truste-box-overlay-inner-height",`${a}px`),l.style.setProperty("--truste-popframe-height",`${a}px`))}};t(n.clientWidth,n.clientHeight);const o=new ResizeObserver(a=>{for(const r of a)requestAnimationFrame(()=>{t(r.contentRect.width,r.contentRect.height),i()})});return o.observe(n),{cleanup:()=>o.disconnect(),reapply:()=>{t(n.clientWidth,n.clientHeight),i()}}}function lc(n,e){const t=new Yl(n).parseAllMediaQueries(),i=ac(t),o=rc(n,t),{cleanup:a,reapply:r}=sc(e,i);return{transformedCSS:o,cleanup:a,reapply:r}}function cc(n,e){let t=0,i=!1,o=!1,a="";for(let r=e;r<n.length;r++){const l=n[r],c=r>0?n[r-1]:"";if((l==='"'||l==="'")&&c!=="\\"){o?l===a&&(o=!1,a=""):(o=!0,a=l);continue}if(!o){if(n.substring(r,r+4)==="url("){i=!0,r+=3;continue}if(i&&l===")"){i=!1;continue}if(!i){if(l==="{")t++;else if(l==="}"){if(t===0)return r;t--}}}}return-1}async function pc(n){if(n.length===0)return;const e=document.head.querySelector("style[data-shadow-fonts]"),t=e||document.createElement("style");e||(t.setAttribute("data-shadow-fonts","true"),document.head.appendChild(t));const i=n.map(r=>r.originalRule).join(`

`),o=t.textContent||"";n.some(r=>!o.includes(r.originalRule.replace(/\s+/g," ")))&&(t.textContent=o+`
`+i)}function dc(n,e){const t=[];let i=n;const o=/@font-face\s*\{/gi;let a;const r=[];for(;(a=o.exec(n))!==null;){const l=a.index,c=a.index+a[0].length-1,d=cc(n,c+1);if(d!==-1){const p=n.substring(c+1,d),u=p.match(/font-family\s*:\s*[^;]+/g),f=p.match(/src\s*:\s*[^;]+/g);if(u&&f&&u.length>1){const g=Math.min(u.length,f.length),x=p.replace(/font-family\s*:\s*[^;]+;?/g,"").replace(/src\s*:\s*[^;]+;?/g,"").replace(/\/\*[\s\S]*?\*\//g,"").replace(/\s+/g," ").trim();for(let b=0;b<g;b++){let h=`${u[b]};
    ${f[b]};`;x&&(h+=`
    ${x}`);const _=`@font-face {
    ${St(h,e)}
}`;t.push({originalRule:_})}}else{const x=`@font-face {${St(p,e)}}`;t.push({originalRule:x})}r.push({start:l,end:d+1,content:""})}}return r.reverse().forEach(l=>{i=i.substring(0,l.start)+l.content+i.substring(l.end)}),{cleanedCss:i,allFontFaces:t}}async function uc(n){const{css:e,containerElement:t,shadowRoot:i,parentBaseUrl:o,cmType:a}=n,r=new Set,l=e.replace(ql,c=>{const d=Xl(c);return d&&r.add(d),""});return await Promise.all([...r].map(c=>{const d=Rr(c)??o;return gc({url:c,containerElement:t,shadowRoot:i,parentBaseUrl:d,cmType:a})})),l}async function gc(n){const{url:e,containerElement:t,shadowRoot:i,parentBaseUrl:o,cmType:a}=n,r=Ir(e,o);await fetch(r).then(l=>l.text()).then(async l=>{const c=Bo(i,Es(e));await Ho({css:l,styleElement:c,containerElement:t,shadowRoot:i,parentBaseUrl:o,cmType:a})})}function fc(n){let e=n;return e=e.replace(/:root\s*{([^}]*)}/g,(t,i)=>`:host{${i}}`),e=e.replace(/:root([\.\:][\w-]+)\s*{([^}]*)}/g,(t,i,o)=>`:host${i}{${o}}`),e=e.replace(/:root(\s+[^{]+)\s*{([^}]*)}/g,(t,i,o)=>`:host${i}{${o}}`),e}let An;function bc(){if(An!==void 0)return An;if(typeof document>"u")return An=16,An;let n;try{n=document.createElement("iframe"),n.setAttribute("aria-hidden","true"),n.tabIndex=-1,n.style.cssText="position:absolute;left:-9999px;top:0;width:0;height:0;border:0;visibility:hidden;",(document.body??document.documentElement).appendChild(n);const e=n.contentDocument,t=n.contentWindow;if(!e||!t)return An=16,An;const i=e.documentElement,o=t.getComputedStyle(i).fontSize,a=Number.parseFloat(o);return An=Number.isFinite(a)&&a>0?a:16,An}catch{return An=16,An}finally{n==null||n.remove()}}function hc(n){const e=bc();return n.replace(/(-?(?:\d*\.)?\d+)rem\b/gi,(t,i)=>`${Number.parseFloat(i)*e}px`)}function mc(n,e,t){let i;if(n==="PRO")i=t?"16px":"100%";else switch(e){case"gdpr_accessible":i="14px";break;case"default_eu":case"default_eu_accessible":case"gdpr":case"iab":case"iab_accessible":case"gpp":default:i="12px";break}return`:host { font-size: ${i}; }`}function xc(n){let e="",t=0,i="normal",o="";for(;t<n.length;){const a=n[t],r=n[t+1]??"",l=n[t-1]??"";switch(i){case"normal":(a==='"'||a==="'")&&l!=="\\"?(i="quote",o=a,e+=a):a==="/"&&r==="*"?(i="comment",t++):e+=a;break;case"quote":e+=a,a===o&&l!=="\\"&&(i="normal",o="");break;case"comment":a==="*"&&r==="/"&&(i="normal",t++);break}t++}return e}function _c(n,e){let t=e;return n==="PRO"&&(t=t.replace(/\.trustarc-overlay-root/g,":host"),t=t.replace(/\.trustarc-two-step-overlay-root/g,":host")),t}async function Ho(n){if(!n.css)return()=>{};const{css:e,styleElement:t,containerElement:i,shadowRoot:o,parentBaseUrl:a}=n,r=xc(e),l=_c(n.cmType,r),{cleanedCss:c,allFontFaces:d}=dc(l,a);d.length>0&&await pc(d);const p=St(c,a),{transformedCSS:u,cleanup:f,reapply:g}=lc(p,i),x=await uc({css:u,containerElement:i,shadowRoot:o,parentBaseUrl:a,cmType:n.cmType}),b=nc(x),h=fc(b),m=hc(h),_=mc(n.cmType,n.layout,n.isAccessibleLayout),z=":host { all: initial; display: block !important; direction: ltr; } .wrapper { all: initial; display: block; direction: ltr; }"+`
`+":host { transform: translate(0); display: block !important; } @supports (contain: paint) { :host { contain: paint; transform: none; } }"+`
`+":host { color: #000000; } .wrapper { color: #000000; }"+`
`+_+`
`+":host { -webkit-font-smoothing: auto; }"+`
`+m;return t.textContent=z,g(),f}function yc({wrapperRef:n,appStyles:e,cmType:t,userCss:i,onStylesInjected:o,deps:a}){const r=y(l=>l.configSlice.loaded);W(()=>{if(!r||!n.current)return;const l=yo(n);if(!l){o==null||o();return}const c=n.current.closest(".wrapper");if(!c){T.warn("Could not find parent wrapper element"),o==null||o();return}let d=!1;const p=[];return(async()=>{try{let f=e;Ut(i)||(f+=`

/* User CSS */

`+i),T.info(`[TrustArc Consent Manager] Injecting ${t.toLowerCase()} app CSS into Shadow DOM`);const g=Bo(l,"data-app-styles"),x=await Ho({css:f,styleElement:g,containerElement:c,shadowRoot:l,cmType:t});x&&p.push(x)}catch(f){T.error(`[TrustArc Consent Manager] Error loading CSS for ${t}:`,f)}})().then(()=>{d||o==null||o()}),()=>{d=!0,p.forEach(f=>f())}},a)}function wc(n,e){W(()=>{if(!e)return;const t=setTimeout(()=>{const i=yo(n);i&&Ts(i,e)},100);return()=>clearTimeout(t)},[e,n])}const vc=`html, body, div, span, applet, object, iframe,
h1, h2, h3, h4, h5, h6, p, blockquote, pre,
a, abbr, acronym, address, big, cite, code,
del, dfn, em, font, img, ins, kbd, q, s, samp,
small, strike, strong, sub, sup, tt, var,
b, u, i, center,
dl, dt, dd, ol, ul, li,
fieldset, form, label, legend,
table, caption, tbody, tfoot, thead, tr, th, td, button {
  background: transparent;
  border: 0;
  margin: 0;
  padding: 0;
  vertical-align: baseline;
}

body { font-size: 12px; font-family: "Source Sans Pro", sans-serif; line-height: 20px; }
body.main { background: url(asset/bg.png) no-repeat center 0; line-height: 20px; }
body.pbg { background: #fff url(asset/pbg.jpg) repeat-y 1px 0; }
input, textarea, select { font-size: 12px; font-family: 'Lucida Grande', Arial, Helvetica, sans-serif; }



/***INDEX.HTML***/
.mainheader {  container-type: inline-size;  }
.mainHeader h1 { color: #2C2D31; font-size: 18px; display: inline-block; }
.accept-decline-buttons { float: right; }
#accept_all_button{ background: no-repeat scroll 0 0 #5fc323;padding: 7px 10px 7px 14px;border: 1px solid #5fc323;text-shadow: none;
  color: #ffffff;border-radius: 4px;font-family: Arial, Tahoma;display: none;font-size: 14px;font-stretch: normal;font-style: normal;
  font-weight: bold;letter-spacing: normal;line-height: normal;text-align: center;text-transform: uppercase;}
#decline_all_button{background: no-repeat scroll 0 0 #5fc323;padding: 7px 10px 7px 14px;border: 1px solid #5fc323;text-shadow: none;color: #ffffff;
  border-radius: 4px;font-family: Arial, Tahoma;display: none;font-size: 14px;font-stretch: normal;font-style: normal;font-weight: bold;
  letter-spacing: normal;line-height: normal;text-align: center;text-transform: uppercase;}

.prefHeader { border-bottom: 1px solid #ddd; padding-bottom: 10px; }
.prefHeader img { float: right; vertical-align: middle; }

.mainContent { padding: 20px 20px 0px 20px; position: relative; }
.mainContent, .consentToAll { line-height: 18px; color: #666; background: #fff; border-top: 1px solid #fff; }
.mainContent p, .consentToAll p { margin: 7px 0; }
.mainContent button, .consentToAll a { text-decoration: none; }
.mainTitle2 { margin: 0 0 20px 0; color: #669B00; font-size: 14px; font-family: 'Abel'; font-weight: normal; text-transform: uppercase; }
.mainContent .buttons, .consentToAll .buttons { margin-bottom: 0; margin-top: 13px; }

.mainInner { padding: 10px; border: 1px solid #ddd; background: #fcfcfc; overflow: hidden; }
.mainInner { -moz-border-radius: 2px; -webkit-border-radius: 2px; border-radius: 2px; -moz-box-shadow: inset 0 0 3px #e7e7e7; }

#notice p { font-size: 11px; line-height: 15px; }
#consent { padding: 10px 0; font-size: 12px; }
#consent p:first-child { margin-top: 0; }

.consent { padding: 20px 0; }
.contentfield { padding: 15px 0; }

.categoryRadioButton {font-size: 18px;}
.warning { color: red !important; }
.warningMessageHttps { margin: 11px 0px !important; }
/***POP UP***/
.TRUSTe_Overlay { position: fixed; left: 0; top: 0; width: 100%; min-height: 100%; background: #000; opacity: 0.50; display: none; }
.TRUSTe_PopParent { position: fixed; top: 30%; left: 50%; display: none; }
.TRUSTe_PopContainer {
  position: absolute; left: -325px; top: -116px; width: 650px; background: url(asset/trans.png);
  -moz-border-radius: 5px; -webkit-border-radius: 5px; border-radius: 5px; padding: 10px;
}
.TRUSTe_PopInnerContainer { /** left: -391px top: -116px; width: 800 */
  background: #fff; overflow: hidden; -moz-border-radius: 2px; -webkit-border-radius: 2px; border-radius: 2px;
}
.TRUSTe_PopIFrame { border: 0; width: 100%; height: 410px; }

.leftPanel { float: left; width: 280px; border-right: 1px solid #ddd; }
.rightPanel { margin-left: 281px; }

.dtabmenu { list-style: none; border-bottom: 1px solid #ddd; overflow: hidden; margin-bottom: 20px; }
.dtabmenu li { display: inline-block; width: 50%; float: left; }
.dtabmenu li a { display: block; padding: 5px; text-align: center; text-transform: uppercase; font-weight: bold; font-size: 11px; color: #ccc; }
.dtabmenu li a:hover { background: #f7f7f7; color: #666; cursor: pointer; }
.dtabmenu li a.current { background: #ddd; color: #333; }

/***PREF PANEL***/
.prefPanel { margin: 20px 0 0 0; }
.prefPanel .left { float: left; width: 45%; margin-right: 5%; position: relative; }
.prefPanel .right { float: left; width: 44%; padding-left: 30px; border-left: 1px solid #eee; }

/* .prefPanel .cookiecat a.viewcookie { display: inline-block; background: url(asset/toggle.png) no-repeat center center; padding: 18px 15px; } */
/* .prefPanel .cookiecat a.viewcookie:hover { background-color: #eee; } */
/* .prefPanel .cookiecat h3 { margin-left: 30px; } */
/* .prefPanel .cookiecat span.greenbold { color: #1c9122; font-weight: bold; margin-top: 10px; } */

.prefPanel .cookiecat a.viewcookie { cursor: pointer; }
.prefPanel .cookiecat button.viewcookie { cursor: pointer; }
.prefPanel .cookiecat { padding: 10px; background: #fcfcfc; border: 1px solid #ddd; margin-bottom: 10px; -moz-border-radius: 2px; -webkit-border-radius: 3px; border-radius: 3px; }
.prefPanel .cookiecat { -moz-border-radius: 2px; -webkit-border-radius: 2px; border-radius: 2px; -moz-box-shadow: inset 0 0 3px #e7e7e7; }
.prefPanel .cookiecat h3 { margin: 0; font-size: 14px; font-weight: normal; color: #069; text-transform: uppercase; font-family: 'Abel', 'Helvetica Neue',Helvetica,Arial,sans-serif; }
.prefPanel .cookiecat span { font-size: 11px; font-weight: normal; text-transform: none; display: block; color: #666; }
.prefPanel .cookiecat a { font-size: 10px; text-transform: uppercase; display: inline-block; margin-top: 15px; background: url(asset/arrow.png) no-repeat right 2px; padding-right: 10px; color: #666; }
.prefPanel .cookiecat a.expandcookie { background: url(asset/arrow-off.jpg) no-repeat right 2px; background-size: 10px; }
.prefPanel .cookiecat a:hover { cursor: pointer; text-decoration: underline; }
.prefPanel .cookiecat span.greenbold { color: #1c9122; font-weight: bold; margin-top: 10px; }
.prefPanel .cookiecat .loading { display:inline-block; margin-top:15px; padding-right:10px; }

.prefPanel .cookiecatdeactivated h3 { margin: 0; font-size: 14px; font-weight: normal; color: #a9a9a9; text-transform: uppercase; font-family: 'Abel', 'Helvetica Neue',Helvetica,Arial,sans-serif; }

.prefTable {border-collapse: collapse;width: 100%;margin-top: 20px;background: #fff;border-top: 1px solid #ddd;}
.prefTable thead tr:first-child th { font-weight:bold; background-color:#eee; }
.prefTable thead tr:last-child th { border-color: #ddd; }
.prefTable thead tr th:first-child { border-left: 1px solid #ddd; }
.prefTable thead tr th:last-child { border-right: 1px solid #ddd;}
.prefTable thead tr th { border-bottom: 1px solid #eee; vertical-align: middle; padding: 5px; text-align: left; }
.prefTable thead tr td { border-bottom: 1px solid #eee; vertical-align: middle; padding: 5px; text-align: left; background-color:#eee;}
.prefTable tr td { border-bottom: 1px solid #eee; vertical-align: middle; padding: 5px; }
.prefTable tr td a { margin: 0 !important; background: none !important; text-transform: none !important; font-size: 11px; color: #069 !important; }
.prefTable tr td .switch { margin: 0; }
.prefTable tr td .linkoff { margin: 0; width:16px; float:right}
.prefTable tr td:nth-child(3) {word-wrap: break-word;word-break: break-all;}
.prefTable tr td:first-child { border-left: 1px solid #ddd; }
.prefTable tr td:last-child { border-right: 1px solid #ddd; }
.prefTable tr:last-child td { border-color: #ddd; }
.prefTable .whyCantIOptOut a { font-size:12px; padding-right: 0px !important; }
.prefTable .hiddenSeeMore { display:none !important; }

.collapse { display: inline-block; width: 11px; height: 11px; cursor: pointer; }
.colplus { background: url(asset/plus.png) no-repeat 0 -11px; }
.colplus:hover { background-position: 0 0; }
.colminus { background: url(asset/minus.png) no-repeat 0 -11px; }
.colminus:hover { background-position: 0 0; }

.prefTable .whyCantIOptOut a { font-size:12px; padding-right: 0px !important; }
.hideText {width:400px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;}

.truncatedCat {
  /* hide text if it more than N lines  */
  overflow: hidden;
  /* for set '...' in absolute position */
  position: relative;
  /* use this value to count block height */
  line-height: 1.0em !important;
  /* max-height = line-height (1.2) * lines max number (3) */
  max-height: 7.0em;
  /* fix problem when last visible word doesn't adjoin right side  */
  text-align: left;
  /* place for '...' */
  margin-right: -1em;
  padding-right: 1em;
}
/* create the ... */
.truncatedCat:before {
  /* points in the end */
  content: '...';
  /* absolute position */
  position: absolute;
  /* set position to right bottom corner of block */
  right: 0;
  bottom: 0;
}
/* hide ... if we have text, which is less than or equal to max lines */
.truncatedCat:after {
  /* points in the end */
  content: '';
  /* absolute position */
  position: absolute;
  /* set position to right bottom corner of text */
  right: 0;
  /* set width and height */
  width: 1em;
  height: 1em;
  margin-top: 0.2em;
  /* bg color = bg color under block */
  background: white;
}

@media screen and (max-width: 767px){
  .sliderWrapper { left: 7px !important; }
}

@media screen and (max-width: 630px){
  .hideText { display:none; }
  .prefTable .hiddenSeeMore { display: unset !important; }
}

@media (max-width: 580px) {
  .column-headers .consentHeader {
    padding-left: 150px;
    padding-right: 20px;
    text-align: center;
    flex-grow: 1;
  }

  .iab_accessible_managePanel .column-headers {
    align-content: center !important;
    align-items: center !important;
    justify-content: space-between !important;
    padding-left: 50px !important;
    text-align: center !important;
    border-radius: 6px 6px 0 0 !important;
    padding-right: 15px !important;
  }
}

.optoutColumn { float:right; }

.switch { float: right; margin: 10px 5px 0 0; -moz-box-shadow: 1px 1px 0 #fff; -webkit-box-shadow: 1px 1px 0 #fff; box-shadow: 1px 1px 0 #fff; }
.switch span { float: left; display: inline-block; border: 1px solid #bbb; padding: 3px 5px; background: url(asset/switchbg.png) repeat-x 0 -20px; cursor: pointer; }
.switch span.on { border-right:0; -moz-border-radius: 2px 0 0 2px; -webkit-border-radius: 2px 0 0 2px; border-radius: 2px 0 0 2px; }
.switch span.off { -moz-border-radius: 0 2px 2px 0; -webkit-border-radius: 0 2px 2px 0; border-radius: 0 2px 2px 0; }
.switch span.active { color: #088adc; text-shadow: 0 0 3px #96c9fd; background: url(asset/switchbg.png) repeat-x 0 0; }
.switch span.disable { opacity:0.6; filter:alpha(opacity=60); cursor: default; }

.switchtable { position: relative; float: right; top: 0; right: 0; }

.switch { position: inherit; top: -3px; right: 2px; margin: 10px 5px 0 0; -moz-box-shadow: 1px 1px 0 #fff; -webkit-box-shadow: 1px 1px 0 #fff; box-shadow: 1px 1px 0 #fff; }


.options { display: inline-block; float: left; margin-left: 10px; width: 260px; }
.options>div { background: url(asset/line2.png) no-repeat 0 9px; padding-left: 70px; padding-top: 3px; min-height: 20px; min-width: 20px; }
.options div h2 { font-size: 16px; text-transform: uppercase; color: #bbb; font-weight: normal; font-family: 'Abel'; }
.options div p { font-size: 11px; color: #ccc; margin-top: 5px !important; line-height: 14px; }
/**Custom Happy Studio**/
.options div.noAdvertising { background: white; padding-left: 70px; padding-top: 3px; min-height: 20px; min-width: 20px; }

.options div.active h2 { color: #339403; }
.options div.active p { color: #666; }

/***RIGHT PANEL***/
.required, .functional { display: none; }
.required h3, .functional h3, .advertising h3 { font-size: 16px; font-family: 'Abel'; font-weight: normal; color: #007ac7; }
.required h3.not, .functional h3.not, .advertising h3.not { color: #fa3c09; }

.right ul { list-style: none; margin: 10px 0 0 0; }
.right ul li { background: url(asset/list.png) no-repeat 0 10px; display: block; padding-left: 15px; }
.right h3 { font-size: 16px; font-family: 'Abel'; font-weight: normal; color: #007ac7; }
.right h3.not { color: #fa3c09; }

.loading { }
.loading img { vertical-align: middle; }
.loading span { margin-left: 10px; }
#loading2 { position: absolute; top: 40%; width: 95%; text-align: center; z-index: 99999; }

/***MAIN PANEL***/
.mainPanel { line-height: 18px; color: #666; overflow: hidden; }
.mainInner, .submission, .shortNoticePanel { padding: 20px; }
.mainTitle { font-size: 18px; color: #fa7f06; font-weight: normal; margin: 15px 0 20px 0; }
.mainPanel p, .submission p { margin: 10px 0; }
.mainPanel .small, .submission .small { font-size: 11px; line-height: 16px; }
.mainPanel a, .submission a { color: #069; text-decoration: none; }
.mainPanel a:hover, .submission a:hover { color: #fa7f06; text-decoration: none; }

.mainTitle3 { font-size: 16px; }

/***FOOTER***/
.footer {
  background: none repeat scroll 0 0 #F7F7F7;
  border-top: 1px solid #EEEEEE;
  font-size: 11px;
  margin-top: 20px;
  padding: 10px 20px;
  color: #999; font-size: 11px; overflow: hidden;
}
.footerlinetop { border-top: 1px solid #ddd; padding-top: 2px; }
.footer p.notice { margin: 5px 0; color: #666; font-weight: bold; }
.footer .right { float: right; }
.footer .left { float: left; }
.footer .center { float: left; display: inline-block; line-height: inherit;}
.footer a { text-decoration: none; }
.footer a img { display: inline-block; margin-left: 5px; vertical-align: middle; }

.language-selector {}

/***SLIDER***/
.sliderImage { height:20px; width:21px; border-radius: 100%;}
.sliderWrapper { padding: 0px 20px; display: inline-block; position: absolute; left: 16px;  }
.ui-slider-vertical { width: 5px; background: #E2E8F0; position: relative; }
.ui-slider-vertical { -moz-border-radius: 50px; -webkit-border-radius: 50px; border-radius: 50px; }
.ui-slider-vertical.disabled { background: #E0E0E0; cursor: not-allowed; }
.ui-slider-vertical.disabled .ui-slider-handle { pointer-events: none; }
.ui-slider-vertical.disabled .slider-icon { color: #757575; }
.ui-slider-vertical a { width: 19px; height: 19px; background: url(asset/handle.png) no-repeat 0 0; position: absolute; left: -8px; }
.ui-slider-vertical a.ui-slider-handle { margin-bottom: -8px; outline: none; }
.ui-slider-vertical a.ui-state-active { }
.ui-slider-vertical .ui-slider-range { width: 5px; position: absolute; left: 0; -moz-border-radius: 6px 6px 0 0; -webkit-border-radius: 6px 6px 0 0; border-radius: 6px 6px 0 0; }
.ui-slider-vertical .ui-slider-range-min { bottom: 0; }
.ui-slider-vertical .ui-slider-range-max { right: 0; }

.slider-icon {
  color: #000579;
}

.slider-icon:hover,
.slider-icon:active {
  color: #3699F1;
  cursor: pointer;
}

/***CHECK OPTIONS***/
.checkOptions { margin-left: 0; width: auto; }
.checkOptions>div { padding-top: 0; margin-bottom: 20px; position: relative; padding-right: 36px; }
.checkOptions div h2 { padding: 10px; border-left: 1px solid #ddd; margin: 0; }
.checkOptions div p { margin: 0 !important; }
.checkOptions .radioAlignRight .categoryChoice{ position: relative; grid-column: 2; }
.checkOptions div.selected .radioAlignRight .categoryChoice{
  width: 22px;
  height: 22px;
  border: 6px solid #000579;
  border-radius: 50%;
  background-color: white;
  cursor: pointer;
  appearance: none;
  -webkit-appearance: none;
  padding: 0;
  margin: 0;
  display: inline-block;
  vertical-align: middle;
  transition: background-color 0.2s ease, box-shadow 0.2s ease;
  box-sizing: border-box;
}
.checkOptions div.active .radioAlignRight .categoryChoice{
  width: 22px;
  height: 22px;
  border: 6px solid #000579;
  border-radius: 50%;
  background-color: white;
  appearance: none;
  -webkit-appearance: none;
  position: relative;
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.checkOptions div.disabled .radioAlignRight .categoryChoice{
  width: 22px;
  height: 22px;
  border: 2px solid #64748B;
  border-radius: 50%;
  background-color: white;
  appearance: none;
  -webkit-appearance: none;
  position: relative;
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

@media (hover: hover) and (pointer: fine) {
  /* Hover styles only apply on non touch devices */
  .checkOptions div.disabled .radioAlignRight .categoryChoice:hover{
    outline: none !important;
    border: 2px solid #3699F1 !important;
  }

  .checkOptions div.active .radioAlignRight .categoryChoice:hover,
  .checkOptions div.selected .radioAlignRight .categoryChoice:hover {
    outline: none !important;
    box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1 !important;
  }


  .checkOptions div.notClickable .radioAlignRight .categoryChoice:hover{
    cursor: not-allowed;
    border: 2px solid #757575 !important;
  }
}
.checkOptions div.disabled .radioAlignRight .categoryChoice:focus-visible{
  outline: none !important;
  border: 2px solid #000579 !important;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1 !important;
}

.checkOptions div.notClickable .radioAlignRight .categoryChoice{
  width: 22px;
  height: 22px;
  border: 2px solid #757575;
  border-radius: 50%;
  background-color: #E0E0E0;
  appearance: none;
  -webkit-appearance: none;
  position: relative;
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.checkOptions div.active .radioAlignRight .categoryChoice {
  width: 22px;
  height: 22px;
  border: 6px solid #000579;
  border-radius: 50%;
  background-color: white;
  appearance: none;
  -webkit-appearance: none;
  position: relative;
  cursor: pointer;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.checkOptions div.activeDisabled .radioAlignRight .categoryChoice {
  border: 6px solid #757575 !important;
}

.checkOptions div.active .radioAlignRight .categoryChoice:focus-visible{
  outline: none !important;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1 !important;
}

.checkOptions div.active .radioAlignRight .categoryChoice:focus-visible {
  outline: none;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1;
}
.checkOptions .radioAlignRight .categoryChoice{ width: 18px; height: 18px; cursor: pointer; }
.checkOptions div.active h2 { color: #339403; }
.checkOptions div.active span h3, .checkOptions div.active p {color: #1E293B;}

.checkOptions div.selected h2 { color: #339403;}
.checkOptions div.selected span h3, .checkOptions div.selected p {color: #1E293B;}

.checkOptions div.disabled h2 { color: #aaa; border-color: #ccc; }
.checkOptions div.disabled span h3 {color: #64758B;}
.checkOptions div.disabled p { color: #6C757D; }
/**Custom Happy Studio**/
.checkOptions div.noAdvertisingMobile{  background: white !important;}
.checkOptions .accessibleCategoryHeadingName {
  display: grid;
  grid-template-columns: 1fr auto;
  grid-template-rows: auto auto;
  align-items: start;
  column-gap: 8px;
}

.checkOptions .radioAlignRight{
  grid-column: 2;
  grid-row: 1;
  display: flex;
  align-items: center;
}

.checkOptions .accessibleCategoryHeadingName .sliderCatTitle {
  grid-column: 1;
  grid-row: 1;
  margin: 0;
}

.checkOptions .accessibleCategoryHeadingName .sliderCatDef {
  grid-column: 1 / span 2;
  grid-row: 2;
  margin-top: 4px;
}

.checkOptions .highlight { border-left: 1px solid #ccc; padding-left: 10px; display: block; }

/**Radio Button Styling**/
.radioButtonPanel { width: 100% !important; }
.radioButtonBody { gap: 0px !important; padding-bottom: 0px; }
/***BUTTONS***/
.pdynamicbutton { margin: 20px;  }
.pdynamicbutton a { display: inline-block; margin-right: 5px;margin-left: 5px; padding: 8px 15px 8px 35px; line-height: 15px; -moz-border-radius: 3px; -webkit-border-radius: 3px; border-radius: 3px; }
.pdynamicbutton button { display: inline-block; margin-right: 5px;margin-left: 5px; padding: 8px 15px 8px 35px; line-height: 15px; -moz-border-radius: 3px; -webkit-border-radius: 3px; border-radius: 3px; }
/*.pdynamicbutton a { -moz-box-shadow: 1px 1px 3px #a7a6a4; -webkit-box-shadow: 1px 1px 3px #a7a6a4; box-shadow: 1px 1px 3px #a7a6a4; }*/
.pdynamicbutton a { text-align: right;  text-transform: uppercase; font-weight: bold; font-size: 11px; color: #fff; opacity: 0.8 }
.pdynamicbutton button { text-align: right;  text-transform: uppercase; font-weight: bold; font-size: 11px; color: #fff; opacity: 0.8 }
.pdynamicbutton a:hover { text-decoration: none; cursor: pointer; color: #fff; opacity: 1; }
.pdynamicbutton button:hover { text-decoration: none; cursor: pointer; color: #fff; opacity: 1; }

.pdynamicbutton .submit { background: url("asset/gbutton.png") no-repeat 0 0; border: 1px solid #9ba74f; text-shadow: 1px 1px #98ab1b; }
.pdynamicbutton .submit.disable { opacity:0.6; filter:alpha(opacity=60); cursor: default; }
.pdynamicbutton .acceptAllButtonLower {background: no-repeat scroll 0 0 #5fc323;border: 1px solid #5fc323;text-shadow: none;}
.pdynamicbutton .declineAllButtonLower{background: no-repeat scroll 0 0 #5fc323;border: 1px solid #5fc323;text-shadow: none;}
.pdynamicbutton .call { background: url("asset/gbutton.png") no-repeat 0 0; border: 1px solid #9ba74f; text-shadow: 1px 1px #98ab1b; float: left; }
.pdynamicbutton .required { background: url("asset/gbutton_trustarc.png") no-repeat 0 0; border: 1px solid #9ba74f; text-shadow: 1px 1px #98ab1b; float: left; }
.pdynamicbutton .askmelater { background: url("asset/obutton.png") no-repeat 0 0; border: 1px solid #d07d24; text-shadow: 1px 1px #c77d26; display: none; }
.pdynamicbutton .shp { background: url("asset/obutton.png") no-repeat 0 0; border: 1px solid #d07d24; text-shadow: 1px 1px #c77d26; }
.pdynamicbutton .shp2 { font-size: 8px !important; top: 20px !important; }
.pdynamicbutton .advance { background: url("asset/obutton.png") no-repeat 0 0; border: 1px solid #d07d24; text-shadow: 1px 1px #c77d26; }
.pdynamicbutton .cancel { background: url("asset/backbutton.png") no-repeat 0 0; border: 1px solid #a3a3a3; text-shadow: 1px 1px #adadad; }
.pdynamicbutton .close { background: url("asset/graybutton.png") no-repeat 0 0; border: 1px solid #a3a3a3; text-shadow: 1px 1px #adadad; }

.acceptallbutton {
  -moz-border-radius: 4px;
  -webkit-border-radius: 4px;
  border-radius: 4px;
  background-image: linear-gradient(to left, #5fc323, #43b400);
  color: #ffffff !important;
  font-family: Arial, Tahoma;
  cursor: pointer;
  font-size: 14px;
  font-stretch: normal;
  font-style: normal;
  float: right;
  font-weight: bold;
  letter-spacing: normal;
  line-height: normal;
  padding: 15px;
  text-align: center;
  text-shadow: none !important;
  text-transform: uppercase;
  float: right;
  margin-inline-end: 50px;
  align-self: center;
  margin-top: 10px;
}

.acceptallbuttondisabled{
  opacity: 0.6;
  cursor: not-allowed;
}

.acceptallbutton:hover:enabled {
  text-decoration: none;
  cursor: pointer;
  background-color: rgba(0,0,0,0.1);
}

/***CUSTOM STYLE***/
.slimScrollDiv { padding-bottom: 2px; }
.breadcrumbs { font-size: 10px; display: block; margin-bottom: 8px; color: #999; border-bottom: 1px dotted #eee; }
.breadcrumbs a { color :#666; }
.backprefmenu { cursor: pointer; }
.notiSuccess { background: #f0fbdb; padding: 5px 10px; border: 1px solid #8ecd15; -moz-border-radius: 2px; -webkit-border-radius: 2px; border-radius: 2px; }
.show { display: block; }
.floatRight { float: right; }
.noMarginRight { margin-right: 0; }

.highlight {
  color: #136C34;
  border-radius: 6px;
  background-color: #E4F8F0;
  padding: 10.5px;
  display: flex;
  align-items: center;
  gap: 7px;
}

.undoMessage {
  color: #64758B;
}

#debuggerContainer {
  height: 200px;
  overflow-x: hidden;
  overflow-y: auto;
}

/***Accessibility Common Styles***/
/* Accessibility Link Styles */
a {
  font-size: 14px;
  line-height: 1.5;
  color: #1D4ED8;
  border-radius: 4px;
}

a:hover {
  text-decoration: none;
  color: #1D4ED8;
}

a:focus-visible {
  outline: none;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1;
}





.accesibilityList { list-style-position:inside }
.accessibility { width:100% !important; }
.accessibility h2 { border-left:0px !important; }
.accessibility p { border-left:0px !important; }
.accessibility ul{ padding-left:30px; }
.accessibility h3{ padding-left:10px; }
.accessibility .disabled h2{ text-decoration:line-through; }
.hidden { display:none; }

.disabledSubmitMessageAccessible {
  display: flex;
  flex-direction: row;
  align-items: center;
  gap: 7px;
  padding: 10.5px;
  background-color: #FFF2E2;
  color: #816204;
  border-radius: 6px;
  flex-basis: 100%;
}

#button-warning-messages.disabledSubmitMessageAccessible .disableSubmitMessage {
  color: #816204;
}

.warningIconContainer {
  height: 14px;
  width: 14px;
  display: flex;
}

.domainTable td:nth-child(3){
  display:block !important;
}

/*** Granular Scroll CSS **/
.gsHeader{
  position:fixed;
  right:0; top:0;
  width: 100%;
  height: 65px;
  line-height: 65px;
  padding-bottom: 0px!important;
  box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.1);
  -moz-box-shadow:  0 2px 5px 0 rgba(0, 0, 0, 0.1);
  -webkit-box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.1);
  z-index: 88;
}

.gsHeader>h1{
  margin-left: 14px;
}


.gsWarningMsg{
  position:fixed;
  top: 65px!important;
  display:block;
  background-color:#e77d15;
  padding: 10px 47px 10px 15px;
  z-index: 88;
  width: 100%;
}

.gsWarningMsg div{
  color: white !important;
  margin-right: 85px;
}

.gsWarningDivider{
  width: auto;
  display:block;
  height: 1px;
  margin: 10px 40px 10px 10px !important;
  opacity: 0.25;
  border: solid 0.5px #ffffff;
}

.gsWarningMsgClose{
  color: white!important;
  cursor: pointer;
  background: url(asset/ic-close-white.svg);
  width: 9px;
  height: 9px;
  position: absolute;
  top: 10px;
  right: 80px;
}

.gsWarningBtn{
  background: url(asset/ic-warning.svg);
  display:inline-block;
  position:relative;
  margin-left: 6px;
  width: 18px;
  height:16px;
  cursor: pointer;
}

.gsPrefPanel{
  margin-top:65px!important;
  margin-bottom: 100px!important;
  overflow-y: auto!important;
  overflow-x: hidden!important;
  position:fixed; top:0; left:0; right:0; bottom:0 !important;
  padding: 10px 15px 0px 15px !important;
}

.gsContent{
  padding: 0px!important;
}

.gsSubmitContainer{
  bottom: 43px !important;
  width:100% !important;
  position:fixed !important;
  padding-top: 11px;
  margin: 0px !important;
  box-shadow: 0 -5px 5px -5px rgba(0, 0, 0, 0.1);
  -moz-box-shadow:  0 -5px 5px -5px rgba(0, 0, 0, 0.1);
  -webkit-box-shadow: 0 -5px 5px -5px rgba(0, 0, 0, 0.1);
}

.gsSubmit{
  margin: 0px 15px 0px 15px !important;
  padding: 11px 15px 11px 15px !important;
  line-height: 18px !important;
  text-align: center !important;
  display: block !important;
}

.gsOptedOutMsg{
  float:left;
  margin-left: 15px;
}

div.gsOptedOutMsg .highlight{
  font-size: 17px;
  color: #004676;
}

.gsSubmitOptedOut{
  float:right;
}

.gsFooter{
  position:fixed!important;
  right:0!important;
  bottom:0!important;
  width: 100%!important;
  height: 34px !important;
  padding: 0px!important;
  margin: 0px!important;
}

.gsFooter>.right, .gsFooter>.left{
  margin-top: 6px;
}

.gsFooter>.right{
  margin-right: 15px;
}

.gsFooter>.left{
  margin-left: 15px;
}

.gsLogo{
  display:flex !important;
  float:right !important;
  height: 55px !important;
  margin-top: 5px !important;
  margin-right: 15px !important;
  position: static !important;
}

.gsLogo img{
  max-width: 160px !important;
  object-fit: contain !important;
}

.warningTooltip{
  visibility:hidden;
  left: -210%;
  position:absolute;
  top: 20px;
  text-align: center;
  width: 120px;
  line-height: 11px;
  font-size: 8px;
  z-index: 90;
  border: solid 1px #ececec;
  border-radius: 2px;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.09);
  -moz-box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.09);
  -webkit-box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.09);
  background-color: #fcfcfc;
  color: #ababab;
}

.gsWarningBtn:hover .warningTooltip {
  visibility: visible !important;
}

.twoButtonPresent {
  position: relative;
}

.twoButtonPresent > button {
  width: 49%;
}

.twoButtonPresent button:last-of-type {
  position: absolute;
  right: 0;
}

.mainHeaderContainer {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.twoButtonPresent .mainHeaderContainer,
.oneButtonPresent .mainHeaderContainer {
  margin-bottom: 14px;
}

.oneButtonPresent > button {
  width: 100%;
}

@media only screen and (max-width: 780px) {
  .gsHeader{
    line-height:0px;
    height: 90px;
    text-align: center;
    z-index: 888;
  }

  .gsHeader .gsLogo{
    float: none !important;
    margin-top: 10px;
    height: 35px !important;
    margin-right: 0px !important;
    justify-content: center !important;
    width: auto !important;
  }

  .gsHeader h1{
    margin-left: 0px;
    font-size: 3vw !important;
    margin-top: 24px!important;
    display:inline-block !important;
  }

  .gsPrefPanel{
    margin-top:90px !important;
  }

  .gsPrefPanelSubmitted{
    margin-bottom: 130px !important;
  }

  .gsWarningMsg{
    top: 90px !important;
  }

  .gsOptedOutMsg{
    float:none;
  }

  .gsSubmitOptedOut{
    float:none;
  }

  .warningTooltip{
    left: -300%;
  }
  .accept-decline-buttons {
    float: none;
    display: block;
    margin-top: -30px;
  }

  #accept_all_button{
    display: none;
    width: 49%;
  }
  #decline_all_button{
    display: none;
    width: 49%;
  }
}

@media only screen and (max-width:406px){

  .gsPrefPanelSubmitted{
    margin-bottom: 147px !important;
  }

  .gsHeader h1{
    font-size: 5vw !important;
  }

  .gsOptedOutMsg{
    font-size: 3vw;
  }

  div.gsOptedOutMsg .highlight{
    font-size: 4vw;
  }

  .warningTooltip{
    left: -320%;
  }
}

.gsHeaderNoLogo{
  height: 65px;
  line-height: 65px !important;
}

.gsHeaderNoLogo h1{
  margin-top: 0px !important;
}

.gsPrefPanelNoLogo{
  margin-top: 65px !important;
}

.gsWarningMsgNoLogo{
  top: 65px !important;
}

.hiddenElement{
  visibility: hidden;
}


/*** Granular Scroll Advanced CSS ***/
.gsaHeader{
  padding: 0px !important;
}

.gsaSubmitContainer{
  margin-left: 15px;
  margin-right: 15px;
}

.gsaCancel{
  margin-left: 10px;
}

.gsaPrefPanel+.optedOutMessage{
  visibility: hidden;
}

@media only screen and (max-width: 780px) {
  .gsaLogo{
    top: 10px !important;
  }

  .gsaPrefPanel{
    margin-bottom: 194px !important;
  }
}

.advanceSettingAnchor {
  cursor: pointer;
}

.twoStep {
  margin: -20px -20px 0 -20px;
}

.twoStepBottom {
  display: grid;
  column-gap: 2.5%;
  padding-left: 2.5%;
  padding-right: 2.5%;
  padding-top: 12px;
  grid-auto-flow: column;
  height: 44px;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 12px;
  line-height: 30px;
  /* identical to box height */
  text-align: center;

  box-shadow: 0 -1px 4px rgba(0,0,0,0.1);
  background: #FFFFFF;
}

.twoStepHeader {
  height: 84px;
  display: flex;
  box-shadow: 0 2px 4px 0 rgba(0,0,0,0.07);
  justify-content: center;
  align-items: center;
}

.twoStep .twoStepBody {
  height: 175px;
  display: block;
  text-align: center;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 14px;
  line-height: 94px;
}

.accessible .twoStepBody,
.accessible .twoStepBottom{
  font-family: Source Sans Pro;
}

.accessible .twoStepBottom {
  display: flex;
  flex-direction: column;
  gap: 14px;
  height: unset;
}

.accessibleTwoStepButtonContainer > button > img {
  width: 24px;
  height: 24px;
}

@media (min-width: 768px) {
  .accessible .twoStepBottom {
    flex-direction: row;
  }
}

.accessible .twoStepBack,
.accessible .twoStepNo,
.accessible .twoStepYes {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
  padding: 10.5px 17.5px;
  font-size: 14px;
  font-weight: 600;
  border: solid 1px #000579;
  border-radius: 40px;
  text-align: center;
  align-items: center;
  display: flex;
  gap: 7px;
  justify-content: center;
  cursor: pointer;
  outline: none;
  font-family: "Source Sans Pro", sans-serif;
  height: unset;
  flex: 1 1 45%;
}

/* CCM-8972: Provide a visible keyboard-focus indicator for the two-step opt-in
   buttons (Back / No / Yes). They set \`outline: none\` above but, unlike the other
   accessible buttons, had no replacement, so focus was not perceivable when
   tabbing. Matches the focus-ring convention used in the "Focus Outline" block
   below. The buttons sit inside \`.accessibleMainContent\` (21px padding) so the
   ring is not clipped/obscured. WCAG 2.4.11 Focus Not Obscured / 2.4.7 Focus Visible. */
.accessible .twoStepBack:focus-visible,
.accessible .twoStepNo:focus-visible,
.accessible .twoStepYes:focus-visible {
  outline: none;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1;
}

.accessible.twoStep {
  margin: 1px 1px 0 1px;
  position: relative;
}

.accessible .twoStepHeader {
  box-shadow: none;
  height: unset;
  align-items: unset;
  justify-content: unset;
}
.accessible .twoStepHeader h2 {
  font-weight: 600;
  font-size: 21px;
  line-height: 31.5px;
  color: #1E293B;
}

.accessible .twoStepHeader #trustarc-internal-close-button {
  position: absolute;
  right: 0;
  top: 0;
}

.accessible .twoStepBody {
  margin-top: 21px;
  margin-bottom: 21px;
  height: unset;
  display: flex;
  font-size: 14px;
  font-weight: 600;
  line-height: 21px;
  color: #1E293B;
  flex-direction: column;
  align-items: center;
}

.twoStepBodyMessage {
  width: 100%;
  text-align: left;
  margin-top: 21px;
}

.twoStepBodyMessage p {
  margin: unset;
}

.accessible .twoStepBottom {
  box-shadow: none;
  padding: unset;
}

.accessible .twoStepBottom .twoStepBack {
  background: #FFFFFF;
  color: #000579 !important;
}

.switch .gwt-RadioButton{
  border: none;
  color: black;
  background-color: transparent;
  background-image: none;
}

.switch span.gwt-RadioButton.active{
  color: black;
  background-image: none;
}

.prefPanel .cookiecat button.viewcookie {
  font-size: 12px;
  font-weight: normal;
  font-style: normal;
  font-stretch: normal;
  line-height: normal;
  letter-spacing: normal;
  text-align: left;
  color: #2074e5;
  margin: 10px 0;
  padding-right: 10px;
  background-image: none;
  display: inline-block;
  text-transform: none;
}

.loyaltyMessage {
  display: flex;
  flex-direction: column;
}

.loyaltyMessage .loyaltyMessageBottom .loyaltyMessageNo {
  height: 40px;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;

  background: #1E5883;
  border-radius: 4px;

  color: #FFFFFF;
  cursor: pointer;
}

.loyaltyMessage .loyaltyMessageBottom .loyaltyMessageYes {
  height: 40px;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;

  background: #1E5883;
  border-radius: 4px;

  color: #FFFFFF;
  cursor: pointer;
}


.loyaltyMessageBottom {
  display: grid;
  column-gap: 2.5%;
  padding-left: 2.5%;
  padding-right: 2.5%;
  padding-top: 12px;
  grid-auto-flow: column;
  height: 44px;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 12px;
  line-height: 30px;
  /* identical to box height */
  text-align: center;

  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.04);
}


.loyaltyMessageHeader {
  height: 98px;
  display: flex;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.04);
  justify-content: center;
  align-items: center;
}

.loyaltyMessage .loyaltyMessageBody {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 24px;
  gap: 10px;

  height: 100%;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: 400;
  font-size: 16px;
  line-height: 22px;

  color: #000000;
  margin: auto;
}

.br-clear {
  clear: both;
}

.gppViewSwitchPanel {
  display: flex;
  align-items: center;
}

.gpp_on_switch {
  cursor: pointer;
}

.gpp_off_switch {
  cursor: pointer;
}

.bottom .submit {
  cursor: pointer;
}

.bottom .acceptAll {
  cursor: pointer;
}

.bottom .declineAll {
  cursor: pointer;
}

.viewcookie.active {
  background-color: transparent;
}

gpp_category {
  margin-bottom: 10px;
  padding: 10px;
  color: #1E293B;
  display: block;
  font-size: 16px;
}

.gpp_category > div:first-child {
  min-height: 42px;
  margin-bottom: 15px;
}

.gpp_category .category_active {
  width: 113px;
  height: 44px;
  border-radius: 40px;
  gap: 8px;
  background-color: #E4F8F0;
  color: #136C34;
  text-transform: uppercase;
  display: flex;
  float: right;
  justify-content: center;
  align-items: center;
}

.gpp_category > div {
  display: flex;
  justify-content: space-between;
}

.gpp_category p {
  font-weight: 500;
  line-height: 24px;
}

.gpp_category .gpp_switch {
  border-radius: 40px;
  min-width: 122px;
  flex-direction: row;
  margin-top: unset !important;
}

.gpp_category .gpp_switch span { margin-bottom: unset !important; }

.gpp_category .gpp_switch .gpp_off_switch,
.gpp_category .gpp_switch .gpp_on_switch {
  align-content: center;
  line-height: 1.5;
}

.gpp_category .gpp_switch .active {
  background-color: #000579;
  color: #fff;
}

@keyframes rotate {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.gpp_category .loading {
  animation: rotate 2s linear infinite;
  display: inline-block;
  font-size: 24px;
  margin-right: 10px;
}

.gpp_category .loading.switch {
  display: flex !important;
  float: right;
  margin-right: unset !important;
}


/* ACCESSIBLE STYLING*/

@font-face {
  font-family: "Source Sans Pro";
  font-weight: 400;
  src: url(https://consent.trustarc.com/get?name=SourceSansPro-Regular.woff2) format("woff2");
}

@font-face {
  font-family: "Source Sans Pro";
  font-weight: 600;
  src: url(https://consent.trustarc.com/get?name=SourceSansPro-SemiBold.woff2) format("woff2");
}

@font-face {
  font-family: "Source Sans Pro";
  font-weight: 700;
  src: url(https://consent.trustarc.com/get?name=SourceSansPro-Bold.woff2) format("woff2");
}

/* Main Header */

.gdpr_accessible > *,
.accordion {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  font-family: "Source Sans Pro", sans-serif;
}

.gdpr_accessible p {
  margin: 0px;
  color: #1E293B;
}

.accessibleMainContent {
  padding: 21px;
}

.gdprAccessiblePrefPanel {
  padding-bottom: 21px;
}

.gdprAccessiblePrefPanel,
.accordionContentContainer .prefPanel {
  margin: 0px;
}

.accessibleProgressContainer {
  text-align: left;
  width: 100%;
}

.accessibleProgressContainer span {
  margin-bottom: 10px;
  display: inline-block;
}

.progressBar {
  width: 100%;
  height: 10px;
  background-color: #E2E8F0;
  border-radius: 6px;
  overflow: hidden;
  margin-top: 5px;
}

.progressStatus {
  height: 100%;
  background-color: #E11A77;
  border-radius: 5px 0 0 5px;
}

.accessibleHelp,
.accessibleRedirectLink {
  width: 14px;
  height: 14px;
}

.accessibleContentfield {
  padding: 0px 0px 28px 0px;
  font-size: 16px;
  font-weight: 600;
  line-height: 24px;
  text-align: left;
  color: #1E293B;
  display: flex;
  flex-direction: column;
  gap: 21px;
  padding-bottom: 21px;
}

.accessibleDynamicButton {
  margin: 20px 0px;
}

.accessibleGsOptedOutMsg {
  display: none;
}

.accessibleSwitch {
  display: flex;
  min-height: 40px;
  cursor: pointer;
  font-family: "Source Sans Pro", sans-serif;
  width: 100%;
  gap: 4px;
  background-color: #F5F5F5;
  border-radius: 40px;
  border: 1px solid #64748B;
}

@media (min-width: 576px) {
  .accessibleSwitch {
    width: unset;
  }
  .categorySwitch {
    margin-bottom: 7px;
  }
}

.accessibleSwitch :focus {
  position:relative;
}

.accessibleSwitch button {
  flex: 1;
  background-color: white;
  color: #1e293b;
  font-weight: 600;
  font-size: 14px;
  line-height: 1.5;
  cursor: pointer;
  transition: background-color 0.3s, color 0.3s;
  outline-offset: 3px;
  padding: 8px 17.5px;
  font-family: "Source Sans Pro", sans-serif;
  overflow-wrap: break-word;
}

@media (min-width: 576px) {
  .accessibleSwitch button {
    max-width: 85px;
  }
}

.accessibleSwitch .on {
  border-top-left-radius: 40px;
  border-bottom-left-radius: 40px;
  border-style: solid;
  border-color: #64748B;
  border-right: none;
}

.accessibleSwitch .off {
  border-top-right-radius: 40px;
  border-bottom-right-radius: 40px;
  border-style: solid;
  border-color: #64748B;
  border-left: none;
}

.accessibleSwitch button.active {
  background-color: #000579;
  color: white;
  box-shadow: 0 0 0 1px #000579;
}

.truste-switch {
  position: relative;
  display: inline-block;
  width: 42px;
  height: 24.5px;
}

.truste-switch input {
  opacity: 0;
  width: 0;
  height: 0;
}

.truste-slider {
  box-sizing: border-box;
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: flex-start;
  padding: 0 3.5px;
  border: 2px solid #64748B;
  border-radius: 30px;
  cursor: pointer;
  transition: background-color .4s, border-color .4s;
}

.truste-slider:before {
  content: "";
  width: 16.5px;
  height: 16.5px;
  background-color: #64748B;
  border-radius: 50%;
  transition: transform .4s, background-color .4s;
}

.truste-switch input:checked + .truste-slider {
  background-color: #000579;
  border-color: #000579;
}

.truste-switch input:focus-visible + .truste-slider {
  outline: 3px solid #3B82F6;
  outline-offset: 3px;
}

.truste-switch input:checked + .truste-slider:before {
  background-color: white;
  transform: translateX(14.5px);
}

.truste-switch:hover .truste-slider {
  border-radius: 28px;
  border-color: #475569;
}

.truste-switch:hover .truste-slider:before {
  background-color: #475569;
}

.truste-switch input:checked:hover + .truste-slider,
.truste-switch:hover input:checked + .truste-slider {
  background-color: #000466;
  border-color: #000466;
}

.truste-switch input:checked:hover + .truste-slider:before,
.truste-switch:hover input:checked + .truste-slider:before {
  background-color: white;
}

.truste-switch-wrapper {
  display: flex;
  align-items: center;
  gap: 12px;
}

.truste-switch-label {
  font-weight: 600;
  font-size: 14px;
  color: #1e293b;
  line-height: 21px;
  white-space: nowrap;
}

.required-active {
  text-align: center;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
  padding: 10px 32px 10px 32px;
  border-radius: 40px;
  background-color: #E4F8F0;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  box-sizing: border-box;
  width: 100%;
  margin-bottom: 10px;
}

@media (min-width: 576px) {
  .required-active {
    width: unset;
    margin-bottom: unset;
  }
}

.tc-icon {
  display: inline-block;
  background-repeat: no-repeat;
  background-position: center;
  color: white;
}

.tc-icon-chevron-up {
  width: 18px;
  height: 14px;
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2218%22%20height%3D%2214%22%20viewBox%3D%220%200%2018%2014%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%20%3Cpath%20d%3D%22M6.38701%2013C6.28402%2013.0005%206.18197%2012.9804%206.08685%2012.9409C5.99172%2012.9014%205.90544%2012.8433%205.83307%2012.7701C5.68627%2012.6231%205.60382%2012.4238%205.60382%2012.2161C5.60382%2012.0084%205.68627%2011.8091%205.83307%2011.6622L10.505%206.99022L5.83307%202.31827C5.69461%202.16968%205.61922%201.97313%205.62281%201.77005C5.62639%201.56698%205.70866%201.37322%205.85228%201.22959C5.9959%201.08597%206.18966%201.00371%206.39274%201.00012C6.59582%200.996539%206.79236%201.07192%206.94096%201.21039L12.1668%206.43628C12.3136%206.58325%2012.3961%206.78249%2012.3961%206.99022C12.3961%207.19795%2012.3136%207.39718%2012.1668%207.54416L6.94096%2012.7701C6.86859%2012.8433%206.78231%2012.9014%206.68718%2012.9409C6.59206%2012.9804%206.49001%2013.0005%206.38701%2013Z%22%20fill%3D%22%231E293B%22/%3E%20%3C/svg%3E");
}

.tc-icon-chevron-down {
  width: 18px;
  height: 14px;
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2218%22%20height%3D%2214%22%20viewBox%3D%220%200%2018%2014%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%20%3Cpath%20d%3D%22M9.01744%2010.398C8.91269%2010.3985%208.8089%2010.378%208.71215%2010.3379C8.61541%2010.2977%208.52766%2010.2386%208.45405%2010.1641L3.13907%204.84913C3.03306%204.69404%202.98522%204.5065%203.00399%204.31958C3.02276%204.13266%203.10693%203.95838%203.24166%203.82747C3.37639%203.69655%203.55301%203.61742%203.74039%203.60402C3.92777%203.59062%204.11386%203.64382%204.26584%203.75424L9.01744%208.47394L13.769%203.75424C13.9189%203.65709%2014.097%203.61306%2014.2748%203.62921C14.4527%203.64535%2014.6199%203.72073%2014.7498%203.84328C14.8797%203.96582%2014.9647%204.12842%2014.9912%204.30502C15.0177%204.48162%2014.9841%204.662%2014.8958%204.81724L9.58083%2010.1322C9.50996%2010.2125%209.42344%2010.2775%209.32656%2010.3232C9.22968%2010.3689%209.12449%2010.3944%209.01744%2010.398Z%22%20fill%3D%22%231E293B%22/%3E%20%3C/svg%3E");
}

.tc-icon-like {
  width: 24px;
  height: 24px;
  vertical-align: bottom;
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2218%22%20height%3D%2219%22%20viewBox%3D%220%200%2018%2019%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cpath%20d%3D%22M17.2282%207.05C16.798%206.54%2016.1777%206.25%2015.5073%206.25H11.4754V3.5C11.4754%201.98%2010.2448%200.75%208.64404%200.75C7.94371%200.75%207.3134%201.17%207.03327%201.82L4.49204%207.75H2.62114C1.31051%207.75%200.25%208.81%200.25%2010.12V15.89C0.25%2017.19%201.32052%2018.25%202.62114%2018.25H14.1867C15.2772%2018.25%2016.2077%2017.47%2016.3978%2016.39L17.7184%208.89C17.8285%208.23%2017.6484%207.56%2017.2182%207.05H17.2282ZM2.62114%2016.75C2.14091%2016.75%201.75072%2016.36%201.75072%2015.89V10.12C1.75072%209.64%202.14091%209.25%202.62114%209.25H4.23192V16.75H2.62114ZM14.9271%2016.13C14.867%2016.49%2014.5569%2016.75%2014.1867%2016.75H5.74265V8.65L8.41393%202.4C8.45395%202.31%208.54399%202.24%208.73409%202.24C9.42442%202.24%209.97468%202.8%209.97468%203.49V7.74H15.5073C15.7375%207.74%2015.9376%207.83%2016.0776%208C16.2177%208.17%2016.2777%208.39%2016.2377%208.62L14.9171%2016.12L14.9271%2016.13Z%22%20fill%3D%22white%22/%3E%3C/svg%3E");
}

.tc-icon-check {
  width: 24px;
  height: 24px;
  vertical-align: bottom;
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2218%22%20height%3D%2213%22%20viewBox%3D%220%200%2018%2013%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cpath%20d%3D%22M5.99992%2012.75C5.89921%2012.7466%205.80025%2012.7227%205.70911%2012.6797C5.61798%2012.6367%205.53658%2012.5756%205.46992%2012.5L0.469915%207.5C0.374583%207.43819%200.29454%207.35553%200.235828%207.25826C0.177117%207.16099%200.141273%207.05165%200.131001%206.9385C0.120729%206.82535%200.136299%206.71134%200.176535%206.60509C0.216771%206.49884%200.280621%206.40312%200.363264%206.32515C0.445908%206.24718%200.545185%206.18901%200.653599%206.15503C0.762014%206.12105%200.876732%206.11214%200.989093%206.12898C1.10145%206.14582%201.20852%206.18797%201.30221%206.25224C1.3959%206.31651%201.47376%206.40123%201.52992%206.5L5.99992%2010.94L16.4699%200.499999C16.6109%200.408609%2016.7784%200.367188%2016.9457%200.382375C17.113%200.397563%2017.2704%200.468479%2017.3926%200.583759C17.5148%200.699038%2017.5947%200.852004%2017.6196%201.01814C17.6445%201.18428%2017.6129%201.35396%2017.5299%201.5L6.52991%2012.5C6.46325%2012.5756%206.38185%2012.6367%206.29072%2012.6797C6.19958%2012.7227%206.10062%2012.7466%205.99992%2012.75Z%22%20fill%3D%22white%22/%3E%3C/svg%3E");
}

.tc-icon-dislike {
  width: 24px;
  height: 24px;
  vertical-align: bottom;
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2224%22%20height%3D%2225%22%20viewBox%3D%220%200%2024%2025%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cpath%20d%3D%22M18.3707%203.75H6.80067C5.71067%203.75%204.78067%204.53%204.59067%205.61L3.28067%2013.11C3.17067%2013.77%203.35067%2014.44%203.77067%2014.95C4.20067%2015.46%204.82067%2015.75%205.49067%2015.75H9.52067V18.5C9.52067%2020.02%2010.7507%2021.25%2012.3507%2021.25C13.0507%2021.25%2013.6807%2020.83%2013.9607%2020.18L16.5007%2014.25H18.3807C19.6907%2014.25%2020.7507%2013.19%2020.7507%2011.88V6.11C20.7507%204.81%2019.6907%203.75%2018.3807%203.75H18.3707ZM15.2507%2013.35L12.5807%2019.6C12.5407%2019.69%2012.4507%2019.76%2012.2607%2019.76C11.5707%2019.76%2011.0207%2019.2%2011.0207%2018.51V14.26H5.49067C5.26067%2014.26%205.06067%2014.17%204.92067%2014C4.77067%2013.83%204.72067%2013.61%204.76067%2013.38L6.07067%205.88C6.13067%205.52%206.44067%205.26%206.81067%205.26H15.2507V13.35ZM19.2407%2011.88C19.2407%2012.36%2018.8507%2012.75%2018.3707%2012.75H16.7607V5.25H18.3707C18.8507%205.25%2019.2407%205.64%2019.2407%206.11V11.88Z%22%20fill%3D%22white%22/%3E%3C/svg%3E");
}

.tc-icon-cancel-close {
  width: 24px;
  height: 24px;
  vertical-align: bottom;
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2225%22%20height%3D%2225%22%20viewBox%3D%220%200%2025%2025%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%3Cpath%20d%3D%22M13.5601%2012.5L17.9801%208.07996C18.0538%208.0113%2018.1129%207.9285%2018.1539%207.8365C18.1949%207.7445%2018.2169%207.64518%2018.2187%207.54448C18.2205%207.44378%2018.202%207.34375%2018.1642%207.25036C18.1265%207.15697%2018.0704%207.07214%2017.9992%207.00092C17.9279%206.9297%2017.8431%206.87356%2017.7497%206.83584C17.6563%206.79811%2017.5563%206.77959%2017.4556%206.78137C17.3549%206.78314%2017.2556%206.80519%2017.1636%206.84618C17.0716%206.88717%2016.9888%206.94627%2016.9201%207.01996L12.5001%2011.44L8.08012%207.01996C7.93795%206.88748%207.7499%206.81535%207.5556%206.81878C7.3613%206.82221%207.17591%206.90092%207.0385%207.03834C6.90109%207.17575%206.82237%207.36113%206.81895%207.55544C6.81552%207.74974%206.88764%207.93778%207.02012%208.07996L11.4401%2012.5L7.02012%2016.92C6.87967%2017.0606%206.80078%2017.2512%206.80078%2017.45C6.80078%2017.6487%206.87967%2017.8393%207.02012%2017.98C7.16075%2018.1204%207.35137%2018.1993%207.55012%2018.1993C7.74887%2018.1993%207.9395%2018.1204%208.08012%2017.98L12.5001%2013.56L16.9201%2017.98C17.0607%2018.1204%2017.2514%2018.1993%2017.4501%2018.1993C17.6489%2018.1993%2017.8395%2018.1204%2017.9801%2017.98C18.1206%2017.8393%2018.1995%2017.6487%2018.1995%2017.45C18.1995%2017.2512%2018.1206%2017.0606%2017.9801%2016.92L13.5601%2012.5Z%22%20fill%3D%22white%22/%3E%3C/svg%3E");
}

.tc-icon-table-chevron-up {
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2218%22%20height%3D%2214%22%20viewBox%3D%220%200%2018%2014%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%20%3Cpath%20d%3D%22M6.38701%2013C6.28402%2013.0005%206.18197%2012.9804%206.08685%2012.9409C5.99172%2012.9014%205.90544%2012.8433%205.83307%2012.7701C5.68627%2012.6231%205.60382%2012.4238%205.60382%2012.2161C5.60382%2012.0084%205.68627%2011.8091%205.83307%2011.6622L10.505%206.99022L5.83307%202.31827C5.69461%202.16968%205.61922%201.97313%205.62281%201.77005C5.62639%201.56698%205.70866%201.37322%205.85228%201.22959C5.9959%201.08597%206.18966%201.00371%206.39274%201.00012C6.59582%200.996539%206.79236%201.07192%206.94096%201.21039L12.1668%206.43628C12.3136%206.58325%2012.3961%206.78249%2012.3961%206.99022C12.3961%207.19795%2012.3136%207.39718%2012.1668%207.54416L6.94096%2012.7701C6.86859%2012.8433%206.78231%2012.9014%206.68718%2012.9409C6.59206%2012.9804%206.49001%2013.0005%206.38701%2013Z%22%20fill%3D%22%231D4ED8%22/%3E%20%3C/svg%3E");
  background-size: 14px;
  padding: 21px;
  cursor: pointer;
  border-radius: 100%;
}

.tc-icon-table-chevron-down {
  background-image: url("data:image/svg+xml,%3Csvg%20width%3D%2218%22%20height%3D%2214%22%20viewBox%3D%220%200%2018%2014%22%20fill%3D%22none%22%20xmlns%3D%22http%3A//www.w3.org/2000/svg%22%3E%20%3Cpath%20d%3D%22M9.01744%2010.398C8.91269%2010.3985%208.8089%2010.378%208.71215%2010.3379C8.61541%2010.2977%208.52766%2010.2386%208.45405%2010.1641L3.13907%204.84913C3.03306%204.69404%202.98522%204.5065%203.00399%204.31958C3.02276%204.13266%203.10693%203.95838%203.24166%203.82747C3.37639%203.69655%203.55301%203.61742%203.74039%203.60402C3.92777%203.59062%204.11386%203.64382%204.26584%203.75424L9.01744%208.47394L13.769%203.75424C13.9189%203.65709%2014.097%203.61306%2014.2748%203.62921C14.4527%203.64535%2014.6199%203.72073%2014.7498%203.84328C14.8797%203.96582%2014.9647%204.12842%2014.9912%204.30502C15.0177%204.48162%2014.9841%204.662%2014.8958%204.81724L9.58083%2010.1322C9.50996%2010.2125%209.42344%2010.2775%209.32656%2010.3232C9.22968%2010.3689%209.12449%2010.3944%209.01744%2010.398Z%22%20fill%3D%22%231D4ED8%22/%3E%20%3C/svg%3E");
  background-size: 14px;
  padding: 21px;
  cursor: pointer;
  border-radius: 100%;
}

.linksContainer {
  display: flex;
}

.dividerContainer {
  height: 16px;
  width: 36px;
  position: relative;
}

.dividerContent {
  width: 1px;
  height: 16px;
  background-color: #DFE7EF;
  box-sizing: border-box;
  position: absolute;
  top: 80%;
  left: 50%;
  transform: translate(-50%, -50%);
}

.pbold, .gdprAccessibleTable th {
  font-weight: 700;
  text-align: left;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
}

@keyframes spin {
  from {
    transform: rotate(0deg);
  }
  to {
    transform: rotate(360deg);
  }
}

.loadingSpinner {
  -webkit-animation:spin 0.45s linear infinite;
  -moz-animation:spin 0.45s linear infinite;
  animation:spin 0.45s linear infinite;
  float: right;
}

.helperButton,
.redirectButton {
  width: 28px;
  height: 28px;
  border-radius: 28px;
  background-color: transparent;
  display: flex;
  justify-content: center;
  align-items: center;
  position: relative;
  cursor: pointer;
}

.helperTooltipDown,
.defaultEuHelperTooltipDown {
  top: 140%;
}

.helperTooltipUp {
  top: -870%;
}

.defaultEuHelperTooltipUp {
  top: -890%;
}

.redirectHelperTooltipUp {
  top: -250%;
}

.redirectDefaultEuHelperTooltipUp {
  top: -340%;
}

.accessibleTooltip.redirectHelperTooltipUp,
.accessibleTooltip.redirectDefaultEuHelperTooltipUp {
  right: -60% !important;
}

.accessibleTooltip {
  position: absolute;
  background-color: #FFFFFF;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  padding: 17.5px;
  width: 380px;
  color: #1a1a1a;
  display: none;
  gap: 14px;
  flex-direction: column;
  z-index: 9999;
  font-size: 14px;
  right:-62%;
}

.accessibleTooltip p {
  margin: 0;
}

.accessibleTooltip > p:first-of-type {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  display: block;
}

.accessibleTooltip::before {
  content: '';
  position: absolute;
  right: 20px;
  width: 0;
  height: 0;
  border-left: 10px solid transparent;
  border-right: 10px solid transparent;
  border-bottom: 10px solid white;
  z-index: 1;
  top: -10px;
}

.accessibleTooltip::after {
  content: '';
  position: absolute;
  right: 19px;
  width: 0;
  height: 0;
  border-left: 11px solid transparent;
  border-right: 11px solid transparent;
  border-bottom: 11px solid #e0e0e0;
  z-index: 0;
  top: -11px;
}

.accessibleTooltip.helperTooltipUp::before,
.accessibleTooltip.defaultEuHelperTooltipUp::before,
.accessibleTooltip.redirectHelperTooltipUp::before,
.accessibleTooltip.redirectDefaultEuHelperTooltipUp::before,
.accessibleTooltip.showAllTooltipUp::before {
  border-top: 10px solid white;
  border-bottom: none !important;
}

.accessibleTooltip.helperTooltipUp::after,
.accessibleTooltip.defaultEuHelperTooltipUp::after,
.accessibleTooltip.redirectHelperTooltipUp::after,
.accessibleTooltip.redirectDefaultEuHelperTooltipUp::after,
.accessibleTooltip.showAllTooltipUp::after {
  border-top: 11px solid #e0e0e0;
  border-bottom: none !important;
}

.accessibleTooltip.helperTooltipUp::before {
  top: 230px;
}

.accessibleTooltip.helperTooltipUp::after {
  top: 231px;
}

.accessibleTooltip.showAllTooltipUp::before {
  top: 129px;
}

.accessibleTooltip.showAllTooltipUp::after {
  top: 130px;
}

.accessibleTooltip.redirectHelperTooltipUp::before {
  top: 55px;
}

.accessibleTooltip.redirectHelperTooltipUp::after {
  top: 56px;
}

.accessibleTooltip.redirectDefaultEuHelperTooltipUp::before {
  top: 79px;
}

.accessibleTooltip.redirectDefaultEuHelperTooltipUp::after {
  top: 80px;
}

.accessibleTooltip.helperTooltipDown::before {
  top: -10px;
}

.accessibleTooltip.helperTooltipDown::after {
  top: -11px;
}

.accessibleTooltip.showAllTooltipDown::before {
  top: -10px;
}

.accessibleTooltip.showAllTooltipDown::after {
  top: -11px;
}

.accessibleTooltip.defaultEuHelperTooltipUp::before {
  top: 231px;
}

.accessibleTooltip.defaultEuHelperTooltipUp::after {
  top: 231px;
}


.accessibleTooltip ul {
  padding: 17.5px;
  padding-top : 0px;
  margin: 0;
  list-style-type: disc;
  list-style-position: inside;
}

.helperButton.tooltipActive,
.redirectButton.tooltipActive {
  background-color: #D0E1FD;
}

.pSemiBold,
.gdprAccessibleTable td,
.accessibleButtonPrimary,
.accessibleProgressContainer span
.accessibleDomainText,
.accessibleContentfield {
  line-height: 21px;
  font-size: 14px;
  font-weight: 600;
  color: #1E293B;
}

.accessibleHeader h2,
.accessibleCategoryHeadingName h2,
.accessibleHeader .title,
.accessibleHeaderContainer,
.accessibleHeaderContainer h2,
div.accessibleCategoryHeadingName h2,
.accessibleHeaderContainer span {
  color: #1E293B;
  display: inline-block;
  font-size: 21px;
  font-weight: 600;
  line-height: 1.5;
}

.accessibleHeader h2 {
  max-width: calc(100% - 28px);
}

.accessibleHeader .title {
  padding-bottom: 7px;
  display: flex;
  justify-content: space-between;
}

.accessibleHeader .mainHeaderContainer .logo {
  padding: 14px 0px 14px 0px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.accessibleHeader .mainHeaderContainer .logo img {
  max-height: 48px;
}

.accessibleHeader p {
  font-size: 14px;
  font-weight: 600;
}

.text-white {
  color: #FFFFFF;
}

.catDivContainer {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  align-items: flex-start;
  gap: 0;
  width: 100%;
}

.catDivContainer .accessibleSwitch {
  float: right;
}

.categoryDivContent {
  display: flex;
  flex-direction: column;
  gap: 7px;
  margin-bottom: 10px;
}

@media (min-width: 576px) {
  .catDivContainer {
    gap: 42px;
    flex-direction: row;
    padding: 0px 0px 7px 0px;
  }

  .catDivContainer > div:first-child {
    width: 80%;
  }
  .categoryDivContent {
    margin-bottom: unset;
  }
}

.categoryPanelAccessible {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 0px;
}

.accordion.active,
.gpp_accessibility .viewcookie .header.open {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  background-color: #F8F9FA;
}

.accordion {
  display: flex;
  width: 100%;
  padding: 14px;
  gap: 7px;
  border: 1px solid #E2E8F0;
  border-radius: 6px;
  background-color: #F8F9FA;
  color: #334155;
  align-items: center;
  cursor: pointer;
}
@media (max-width: 576px) {
  .accordion,
  .prefTableCatExpand {
    margin-top: 10px;
  }
}

.accordion div {
  color: #334155;
  font-family: 'Source Sans Pro';
  font-size: 14px;
  line-height: 1.5;
}

.divider {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 21px 0px;
}

.divider hr {
  background: #DFE7EF;
  border: 1px solid #DFE7EF;
  height: 1px;
  flex-grow: 1;
  margin: 0;
}

.examplePurposes {
  padding: 0 7px;
}

.prefTableAdvanced .examplePurposes {
  padding: 0;
}

.prefTableAdvanced .examplePurposes p {
  font-weight: 700;
  margin: 0;
  line-height: 1.5;
  font-size: 14px;
}

.prefTableAdvanced .examplePurposes h3 {
  font-weight: 700;
  margin: 0;
  line-height: 1.5;
  font-size: 18px;
}

.categoryDetailsList {
  padding: 7px;
  margin-left: 7px;
}

.categoryDetailsList p {
  margin: 0;
  padding-left: 1em;
  position: relative;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
}

.categoryDetailsList li {
  margin-left: 1em;
  position: relative;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
}

.gpp_accessibility .viewcookie .table .categoryDetailsList span {
  margin: 0;
  padding-left: 1em;
  position: relative;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
}

.prefTableAdvanced .categoryDetailsList {
  padding: 0;
  margin-top: 7px;
  margin-bottom: 14px;
}

.categoryDetailsList p::before {
  content: "•";
  color: black;
  position: absolute;
  left: 0;
  top: 0;
  font-size: 14px;
  line-height: 1.5;
}

.categoryDetailsList span::before {
  content: "•";
  color: black;
  position: absolute;
  left: 0;
  top: 0;
  font-size: 14px;
  line-height: 1.5;
}

.detailsRowContainer {
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.accordionContentContainer {
  width: 100%;
  box-sizing: border-box;
  border: 1px solid rgb(226, 232, 240);
  border-top: none;
  border-bottom-right-radius: 6px;
  border-bottom-left-radius: 6px;
  padding: 14px;
}

.accessibleShowAll {
  vertical-align: middle;
  cursor: pointer;
  anchor-name: --showAll;
}

.showAllTooltipDown {
  right: -0.5%;
  top: 135%;
}

.showAllTooltipUp {
  right: -0.5%;
  top: -620%;
}

.tooltipHeader {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.tooltipHeaderText {
  padding-top: 3px;
}

.tooltipBody {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.accessibleDomainText {
  text-wrap: auto;
  white-space: nowrap;
  overflow: hidden;
  display: inline-block;
  max-width: 180px;
  vertical-align: middle;
  text-overflow: ellipsis;
}

.domainTextContainer {
  width: 200px;
}

.collapseContainer {
  display: flex;
  justify-content: center;
  align-items: center;
}

.collapseTableButton {
  width: 42px;
  padding: 10.5px;
  cursor: pointer;
}

.tableSwitchPanelContainer,
.tableHelpContainer,
.domainPanelContainer,
.tableSwitchPanelContainer,
.tableRedirectContainer {
  width: fit-content;
  position: relative;
}

.gdprAccessibleTable {
  border-collapse: unset;
  border-spacing: 0;
  border-radius: 6px;
  overflow: visible;
  border: 1px solid #E3E8F0;
  table-layout: fixed;
  width: 100%;

  thead {
    background-color: #F8FAFC;
  }

  thead tr th:first-child {
    border-top-left-radius: 6px;
  }

  thead tr th:last-child {
    border-top-right-radius: 6px;
  }

  th {
    text-align: left;
    padding: 7px;
    color: #1E293B;
  }

  .gdprTableAccordion {
    align-items: center;
  }

  td {
    padding: 7px;
    color: #1e293b;
    vertical-align: middle;
    border-top: 1px solid #E2E8F0;
  }

  td[colspan="100"] {
    padding: 14px;
  }
}

.accessibleDynamicButtons {
  display: flex;
  flex-direction: column;
  width: 100%;
  gap: 14px;
}

.tc-btn-icon {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 7px;
}

.accessibleButtonPrimary,
.cancelOptOut,
.messageLoading,
.doneOptOut {
  border: none;
  padding: 10.5px 24.5px;
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: #FFFFFF;
  font-family: "Source Sans Pro", sans-serif;
  box-shadow: 0px 2px 3px -2px #000579;
  border-radius: 40px;
  cursor: pointer;
  gap: 7px;
  color: white;
}

.accessibleButtonPrimary {
  color: white !important;
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 7px;
}

.accessibleButtonPrimary img {
  width: 24px;
  height: 24px;
}

.accessibilityButtonIcon {
  display: flex;
  flex-direction: row;
  align-items: center;
  justify-content: center;
  gap: 7px;
}

/* Disclosure*/

.disclosureTitle {
  color: #1E293B;
  padding-bottom: 21px;
  font-weight: 700;
}

.disclosureBody {
  font-weight: 600;
  font-size: 14px;
  padding: 21px 0px;
  line-height: 21px;
  color: #1E293B;
}

.disclosureBody p {
  margin: 0;
}

.disclosureButtonPanel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

@media (min-width: 768px) {
  .disclosureButtonPanel {
    flex-direction: row;
  }
}

.disclosureAcceptAll {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.disclosureDeclineAll {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.disclosureAskMeLater {
  color: #000579 !important;
}

.disclosureViewCookieSettings {
  color: #000579 !important;
}

.accessibleButton {
  padding: 10.5px 17.5px;
  font-size: 14px;
  font-weight: 600;
  border: solid 1px #000579;
  border-radius: 40px;
  width: 100%;
  text-align: center;
  align-items: center;
  display: flex;
  gap: 7px;
  justify-content: center;
  cursor: pointer;
  outline: none;
  font-family: "Source Sans Pro", sans-serif;
  line-height: 21px;
}

@media (min-width: 768px) and (max-width: 991px) {
  .disclosureButtonPanel {
    flex-direction: row;
    flex-wrap: wrap;
  }
  .buttonPanelNoWrap {
    flex-wrap: nowrap;
  }
  .disclosureButtonPanel .accessibleButton {
    flex: 1 1 45%; /* not 50 cuz of the gap */
  }
}

@media (min-width: 992px) {
  .disclosureButtonPanel {
    flex-direction: row;
  }
}

.disclosureAcceptAll {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.disclosureDeclineAll {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.disclosureAskMeLater {
  color: #000579 !important;
}

.disclosureViewCookieSettings {
  color: #000579 !important;
}

#trustarc-internal-close-button {
  float:right;
  width: 28px;
  height: 28px;
  background-image: url("asset/close-button.svg");
  cursor: pointer;
  border-radius: 100%;
  outline: none;
}

.footerAccessible {
  display: flex;
  flex-direction: column;
  padding: 21px;
  margin: auto;
  font-family: "Source Sans Pro", sans-serif;
}

.footerAccessible .footerRight {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
}

.footerAccessible a[href] {
  color: #1D4ED8 !important;
  text-decoration: underline !important;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 600;
}

.footerAccessible a[href]:hover {
  color: #1D4ED8 !important;
  text-decoration: none !important;
  cursor: pointer;
}

.footerAccessible a[href]:focus-visible {
  outline: none;
  box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
}

.footerAccessible .center {
  width: 100%;
  flex-direction: column;
  display: flex;
}

.languageSelectorContainer {
    display: flex;
    gap: 7px;
    padding-top: 7px;
    align-items: center;
    font-family: "Source Sans Pro", sans-serif;
    font-size: 14px;
    font-weight: 600;
    line-height: 21px;
    color: #1E293B;
}

.languageSelectorAccessible {
    padding: 7px;
    background: #FFFFFF;
    border: 1px solid #757575;
    border-radius: 6px;
    cursor: pointer;
    font-family: inherit;
}

.languageSelectorAccessible:hover {
    border: 1px solid #3699F1;
}

.languageSelectorAccessible:focus-visible {
    border: 1px solid #3699F1;
    box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
}

.languageSelectorAccessible:active {
    border: 2px solid #3699F1;
}

.footerDivider {
  width: 100%;
  height: 0;
  border: 1px solid #DFE7EF;
  background: #DFE7EF;
  margin: 17.5px 0;
}

@media (min-width: 768px) {
  .footerAccessible {
    flex-direction: row;
    width: fit-content;
  }

  .footerAccessible .center {
    flex-direction: row;
    width: fit-content;
  }

  .footerDivider {
    width: 0;
    height: 44px;
    margin: 0 17.5px;
  }
}

.footerLeft {
  margin: auto;
  font-size: 14px;
  font-weight: 600;
  text-decoration-style: solid;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
  text-align: center;
}

.footerRight {
  margin: auto;
}

/* Pref Table */

.prefTableButtonPanel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.prefTableUpperButtonPanel {
  display: flex;
  flex-direction: column;
  margin-bottom: 14px;
  gap: 14px;
}

.accessibleSaveAndExit
.goBack {
  color: #000579 !important;
}

.prefTableSubmit {
  color: #000579;
}

.prefTableSubmit img,
.accessibleSaveAndExit img,
.goBack img {
  /* set to black first to make filter consistent */
  color: #000000 !important;
  filter: invert(12%) sepia(65%) saturate(7293%) hue-rotate(241deg) brightness(49%) contrast(121%);
}

.prefTableSubmitFilled {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.prefTableSubmitFilled img,
.submitPreferencesButtonLower img {
  filter: brightness(0) saturate(100%) invert(100%) brightness(200%) contrast(1000%)
}

.prefTableCancel {
  color: #000579 !important;
}

.prefTableSettings {
  color: #000579 !important;
}

.prefTableAccept {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.prefTableDecline {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.sliderBody {
  display: flex;
  flex-direction: column;
  border-top: 1px solid #DFE7EF;
  padding-top: 31.5px;
  row-gap: 21px;
  padding-bottom: 28px;
}

.sliderLeft {
  display: flex;
  column-gap: 21px;
}

.sliderRight {
  display: flex;
  flex-direction: column;
  row-gap: 11px;
}

@media (min-width: 768px) {
  .sliderBody {
    flex-direction: row;
  }
  .sliderLeft {
    width: 50%;
  }
  .sliderRight {
    width: 50%;
  }
}

.sliderWrapperAccessible .ui-slider-vertical .ui-slider-range {
  background: #000579;
}

.sliderOptions {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.sliderOptions div {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.sliderOptions .inactive,
.sliderOptions .inactive h2 {
  text-decoration-line: line-through;
  color: #64758B;
}

.sliderOptions .inactive .sliderCatDef {
  color: #6C757D;
}

.sliderOptions .active {
  color: #1E293B;
  row-gap: 11px;
}

.sliderCatDef {
  font-size: 16px;
  font-weight: 600;
  margin-top: 7px !important;
  margin-bottom: 11px !important;
}

.sliderAllowed,
.sliderNotAllowed {
  font-size: 18px;
  font-weight: 700;
  color: #1E293B;
}

.sliderList {
  list-style-type: none;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

.sliderAllowedItem {
  background-image: url("asset/green-check.svg");
  background-repeat: no-repeat;
  padding-left: 28px;
  font-family: Source Sans Pro;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
  color:  #136C34;
}

.sliderNotAllowedItem {
  background-image: url("asset/grey-cross.svg");
  background-repeat: no-repeat;
  padding-left: 28px;
  font-family: Source Sans Pro;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
  text-decoration-line: line-through;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
  color: #6C757D;
}


.prefTableAdvanced {
  color: #1E293B;
}

.prefTableAdvanced h2 {
  font-size: 21px;
  font-weight: 600;
  line-height: 1.5;
}

.sliderCatTitle h3, h3.sliderCatTitle {
  font-weight: 700;
  font-size: 18px;
  line-height: 27px;
}

.prefTableCat {
  border-top: 1px solid #DFE7EF;
  padding: 21px 0;
}

@media (min-width: 576px) {
  .prefTableCat .accessibleSwitch {
    float: right;
    width: fit-content;
  }
}

.prefTableCat td .accessibleSwitch {
  float: none;
}

.prefTableCatExpand {
  width: 100%;
  display: flex;
  padding: 17.5px 14px;
  font-size: 14px;
  color: #334155 !important;
  border: 1px solid #E2E8EF;
  background-color: #F8F9FA;
  align-items: center;
  column-gap: 7px;
  cursor: pointer;
  border-radius: 6px;
  outline-offset: 3px;
  font-family: "Source Sans Pro", sans-serif;
  line-height: 21px;
  font-weight: 600;
}

.prefTableActiveLabel {
  padding: 10px 32px;
  border-radius: 40px;
  background-color: #E4F8F0;
  color: #136C34;
  font-size: 14px;
  box-sizing: border-box;
  width: 100%;
  text-align: center;
}

.prefTableCatNameDesc {
  display: inline-block;
}

@media (min-width: 576px) {
  .prefTableActiveLabel {
    float: right;
    margin-bottom: unset;
    width: unset;
  }
  .prefTableCatNameDesc {
    width: 65%;
  }
}

@media (max-width: 576px) {
  table.prefTableTable.withoptout > thead > tr > th:nth-child(4),
  table.prefTableTable.withoptout > tbody > tr > td:nth-child(4),
  table.prefTableTable.withoptout > colgroup > col:nth-child(4) {
    display: none;
  }

  table.prefTableTable.withoptout > thead > tr > th:nth-child(1),
  table.prefTableTable.withoptout > tbody > tr > td:nth-child(1),
  table.prefTableTable.withoptout > colgroup > col:nth-child(1),
  table.prefTableTable.optoutless > thead > tr > th:nth-child(1),
  table.prefTableTable.optoutless > tbody > tr > td:nth-child(1),
  table.prefTableTable.optoutless > colgroup > col:nth-child(1){
    width: 13%;
  }

  table.prefTableTable.withoptout > thead > tr > th:nth-child(2),
  table.prefTableTable.withoptout > tbody > tr > td:nth-child(2),
  table.prefTableTable.withoptout > colgroup > col:nth-child(2) {
    word-wrap: break-word;
    width: 31%;
  }

  table.prefTableTable.withoptout > thead > tr > th:nth-child(3),
  table.prefTableTable.withoptout > tbody > tr > td:nth-child(3),
  table.prefTableTable.withoptout > colgroup > col:nth-child(3) {
    width: 63%;
    padding-left: 7px;
  }
}

.prefTableTable {
  border-collapse: unset;
  border-spacing: 0;
  border-radius: 6px;
  border: 1px solid #E3E8F0;
  table-layout: fixed;
  width: 98%;

  thead {
    background-color: #F8FAFC;
  }

  th {
    text-align: left;
    padding: 7px;
    color: #1E293B;
  }

  th:first-child {
    border-top-left-radius: 6px;
  }

  th:last-child {
    border-top-right-radius: 6px;
  }

  tbody tr {
    border-bottom: 1px solid #E2E8F0;
  }

  tbody tr:last-child {
    border-bottom: none;
  }

  td {
    padding: 7px;
    color: #1e293b;
    vertical-align: middle;
    border-top: 1px solid #E2E8F0;
    line-height: 21px;
    font-size: 14px;
    font-weight: 600;
    border-right: none;
  }
}

.prefTableTable th {
  font-size: 14px;
  font-weight: 700;
  line-height: 21px;
  text-align: left;
  text-underline-position: from-font;
  text-decoration-skip-ink: none;
}

.prefTableTable td[colspan="100"] {
  padding: 14px;
}

.prefTableTable .whyCantIOptOut {
  font-size: 14px;
  display: flex;
  flex-direction: column;
  gap: 7px;
}

@media (min-width: 768px) {

  .prefTableButtonPanel {
    flex-direction: row;
    flex-wrap: wrap;

  }

  .prefTableButtonPanel .accessibleButton {
    flex-basis: 30%;
    flex-grow: 1;
  }

  .prefTableUpperButtonPanel {
    flex-direction: row;
  }

  .prefTableUpperButtonPanel.hasTwoButtons .accessibleButton {
    flex-basis: 49%;
    flex-grow: 1;
  }
}


@keyframes spin {
  100% {
    -webkit-transform: rotate(360deg);
    transform: rotate(360deg);
  }
}

.loadingSpinner {
  -webkit-animation: spin 0.45s linear infinite;
  -moz-animation: spin 0.45s linear infinite;
  animation: spin 0.45s linear infinite;
  float: right;
}

.prefTableCatExpand img {
  transition: transform 0.20s;
}

.prefTableCatExpand.viewcookie img {
  transform: rotate(0deg);
}

.prefTableCatExpand.expandcookie img {
  transform: rotate(90deg);
}

.prefTableCatExpand.expandcookie {
  border-bottom-left-radius: 0;
  border-bottom-right-radius: 0;
  z-index: 1; /* makes the focus outline on top of the border of the div below it */
  position: relative;
}

@media (max-width: 576px) {
  table.gdprAccessibleTable.withoptout > thead > tr > th:nth-child(4),
  table.gdprAccessibleTable.withoptout > tbody > tr > td:nth-child(4),
  table.gdprAccessibleTable.withoptout > colgroup > col:nth-child(4) {
    display: none;
  }

  table.gdprAccessibleTable.withoptout > thead > tr > th:nth-child(2),
  table.gdprAccessibleTable.withoptout > tbody > tr > td:nth-child(2),
  table.gdprAccessibleTable.withoptout > colgroup > col:nth-child(2) {
    width: 31%;
  }

  table.gdprAccessibleTable.withoptout > thead > tr > th:nth-child(3),
  table.gdprAccessibleTable.withoptout > tbody > tr > td:nth-child(3),
  table.gdprAccessibleTable.withoptout > colgroup > col:nth-child(3) {
    width: 63%;
    padding-left: 7px;
  }

  .gdprAccessibleTable .gdprTableAccordion {
      justify-content: unset;
  }

  .accessibleDomainText {
    white-space: break-spaces;
    max-width: 100%;
    overflow-wrap: break-word;
  }
  .domainPanelContainer {
    width: unset;
    max-width: 100%;
  }
}

.prefTableTable .accessibleDomainText {
  white-space: break-spaces;
  max-width: 100%;
  overflow-wrap: break-word;
  padding-bottom: 4px;
}

.prefTableTable .domainPanelContainer {
  max-width: 100%;
}

.showAllContainer {
  max-width: fit-content;
  display: inline-block;
  padding: 0px 2px; /*need this to show the outline for show all*/
}

.prefTableTable .showAllTooltipDown,
.prefTableTable .showAllTooltipUp {
  max-width: 250px;
  width: fit-content;
  position-anchor: --showAll;
  right: anchor(right);
  top: anchor(200%);
  overflow-wrap: break-word;
}

@media (min-width: 768px) {
  .prefTableTable .showAllTooltipDown,
  .prefTableTable .showAllTooltipUp {
    max-width: 300px;
  }
}

@media (min-width: 992px) {
  .prefTableTable .showAllTooltipDown,
  .prefTableTable .showAllTooltipUp {
    max-width: 380px;
  }
}

.showAllAnchor {
  max-width: fit-content;
}

.disclosureAccessible,
.prefTableAccessible,
.loadingAccessible,
.optOutCancelAccessible,
.optOutDoneAccessible,
.gdpr_accessible {
  font-family: "Source Sans Pro", sans-serif;
}

.modal-iab {
  background: white;
  width: 350px;
  border-radius: 8px;
  margin: auto;
  overflow: auto;
  height: 90vh;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.tab-header-iab {
  display: flex;
  flex-direction: column;
  float: left;
  justify-content: space-between;
  cursor: pointer;
  font-size: 16px;
  position: sticky;
  top: 0;
  z-index: 1000;
  padding: 5px 0;
}

.tab-header-iab div {
  flex: 1;
  text-align: center;
  padding: 10px 5px;
  transition: background-color 0.3s;
}

.tab-header-iab div:hover, .tab-header-iab .active {
  background-color: #ddd;
}

.content-iab {
  padding: 20px;
  line-height: 1.6;
  min-height: 70vh; /* Ensure each section can potentially fill the screen */
}

.switch-iab {
  position: relative;
  display: inline-block;
  width: 34px;
  height: 20px;
  float: right;
}

.switch-iab input {
  opacity: 0;
  width: 0;
  height: 0;
}

.slider-iab {
  position: absolute;
  cursor: pointer;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: #ccc;
  transition: .4s;
  border-radius: 20px;
}

.slider-iab:before {
  position: absolute;
  content: "";
  height: 14px;
  width: 14px;
  left: 3px;
  bottom: 3px;
  background-color: white;
  transition: .4s;
  border-radius: 50%;
}

input:checked + .slider-iab {
  background-color: #4CAF50;
}

input:checked + .slider-iab:before {
  transform: translateX(14px);
}

.footer-buttons-iab {
  display: flex;
  justify-content: space-between;
  padding: 10px 20px 20px;
  position: sticky;
  bottom: 0;
  background: #fff;
  width: 100%;
}

.footer-buttons-iab button {
  border: none;
  outline: none;
  padding: 10px;
  border-radius: 4px;
  cursor: pointer;
  background-color: #4CAF50;
  color: white;
}

.cookie-container-iab {
  display: flex;
  flex-direction: column;
}

.displayNone {
  display: none;
}

.optedOutMessageAccessible {
  margin-top: 14px;
  font-size: 14px;
  color: #64758B;
  font-weight: 400;
}

.optedOutMessage div {
  display: flex !important;
  flex-direction: column !important;
  gap: 7px;
}

.warningMessageOptoutFailed {
  color: #816204;
  margin-bottom: 14px;
  padding: 10.5px;
  background-color: #FFF2E2;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  font-weight: 600;
  line-height: 21px;
}

.warningMessageOptoutFailed > div {
  display: flex;
  gap: 7px;
  align-items: center;
}

.thirdPartyWarningPanel {
  margin-left: 21px;
  margin-right: 21px;
}

.prefSubmittedAccessible {
  color: #136C34 !important;
  margin-bottom: 14px;
  padding: 10.5px;
  background-color: #E4F8F0;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 7px;
  font-weight: 600;
  line-height: 21px;
}

.prefSubmittedAccessible img {
  color: #000000 !important;
  filter: invert(29%) sepia(89%) saturate(411%) hue-rotate(90deg) brightness(74%) contrast(94%);
}

.prefTableAccessible {
  /* .maincontent already adds 20px padding*/
  padding: 1px 1px 0px 1px;
}

.prefTableAccessible p {
  font-size: 14px;
  line-height: 21px;
  color: #1E293B;
}

.prefTableAccessible h1 {
  font-size: 24px;
}

/* has to be below .prefTableAccessible p*/
.required-active p {
  color: #136C34;
  margin: 0;
}

.gppMainContent {
  padding: 21px 21px 10.5px;
}

.gpp_accessibility {
  font-family: "Source Sans Pro", sans-serif;
  font-size: 14px;
  color: #1E293B;
  line-height: 1.5;
  font-weight: 600;
}

.gpp_accessibility .modal_header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 21px;
  font-weight: 600;
  line-height: 1.5;
  color: #1E293B;
  padding-bottom: 7px;
}

.gpp_accessibility .modal_header .pref_submitted > h1 {
  font-size: 24px !important;
}

.gpp_accessibility .modal_header img {
  cursor: pointer;
}

.gpp_accessibility .modal_body {
  overflow: auto;
  display: flex;
  flex-direction: column;
  overflow-x: hidden;
  overflow-y: auto;
  position: relative;
  background: #fff;

  scrollbar-color: #BDBDBD; /* For Firefox */
  padding-left: 4px;
  padding-right: 4px;
  margin-left: -4px;
  margin-right: -4px;
}

/* For WebKit browsers (Chrome, Safari, Edge) */
.gpp_accessibility .modal_body::-webkit-scrollbar {
  width: 15px; /* Set the width of the scrollbar */
}

.gpp_accessibility .modal_body::-webkit-scrollbar-track {
  background: white; /* Background color for the scrollbar track */
}

.gpp_accessibility .modal_body::-webkit-scrollbar-thumb {
  background-color: #BDBDBD; /* Background color for the scrollbar thumb */
  border-radius: 14px; /* Border radius for the scrollbar thumb */
  border: 3px solid white; /* Space around the scrollbar thumb */
}

.gpp_accessibility .modal_body::-webkit-scrollbar-button {
  display: none; /* Hide the arrows */
}

.gpp_accessibility .modal_body.no-shadow:after {
  background: unset !important;
}

.gpp_accessibility .gpp_category {
  padding-top: 21px;
  padding-bottom: 21px;
  color: #1E293B;
  display: block;
  font-size: 14px;
}

.gpp_accessibility .gpp_category .category_active {
  width: 113px;
  height: 44px;
  border-radius: 40px;
  gap: 8px;
  background-color: #E4F8F0;
  color: #136C34;
  text-transform: none;
  display: flex;
  float: right;
  justify-content: center;
  align-items: center;
  font-size: 14px;
  line-height: 1.5;
  font-weight: 600;
}

.gpp_accessibility .gpp_category > div {
  display: flex;
  justify-content: space-between;
}

.gpp_accessibility .gpp_category p {
  font-weight: 600;
  line-height: 1.5;
}

.gpp_accessibility .gpp_category .gpp_switch {
  border: 1px solid #64748B;
  border-radius: 40px;
  height: 40px;
  min-width: 122px;
  flex-direction: row;
  margin-top: unset !important;
}

.gpp_accessibility .gpp_category .gpp_switch span {
  margin-bottom: unset !important;
}

.gpp_accessibility .gpp_category .gpp_switch .gpp_off_switch,
.gpp_accessibility .gpp_category .gpp_switch .gpp_on_switch {
  align-content: center;
  width: auto;
}

.gpp_accessibility .gpp_category .gpp_switch .active {
  background-color: #000579;
  color: #fff;
}

.gpp_accessibility .gpp_category .loading {
  animation: rotate 2s linear infinite;
  display: inline-block;
  font-size: 24px;
  margin-right: 10px;
}

.gpp_accessibility .gpp_category .loading.switch {
  display: flex !important;
  float: right;
  margin-right: unset !important;
}

.gpp_accessibility .accessibleHelp {
  width: 14px;
  height: 14px;
  cursor: pointer;
}

.gpp_accessibility .gs_button_close,
.gpp_accessibility .bottom button {
  min-width: 228px;
  width: 100%;
  border-radius: 40px;
  box-shadow: 0px 2px 3px 0px rgba(0, 0, 0, 0.2);
  border: 2px solid transparent;
  font-size: 14px;
  line-height: 1.5;
  font-weight: 600;
  height: 45px;
  font-family: "Source Sans Pro", sans-serif;
  text-transform: none;
  margin-bottom: 0;
}

.footerAccessibleContainer a:focus-visible,
.gpp_accessibility *:focus-visible,
.gpp_accessibility button:focus-visible,
.disclosureBody a:focus-visible,
.sliderImage:focus-visible {
  outline: none;
  box-shadow: 0px 0px 0px 1px rgba(255, 255, 255, 1), 0px 0px 0px 4px #3699F1;
  z-index: 1;
  position: relative;
}

.disclosureBody a:focus-visible {
  border-radius: 4px;
}

.gpp_accessibility .bottom button img {
  margin-right: 10px;
  vertical-align: middle;
}

.gpp_accessibility .gs_button_close {
  cursor: pointer;
  width: 100%;
  background: linear-gradient(to bottom, #2025A7, #000579);
  color: #fff;
  display: flex;
  justify-content: center;
  align-items: center;
  height: 45px;
  font-weight: 600;
  font-size: 16px;
  line-height: 24px;
}

.gpp_accessibility .gs_button_close a {
  text-decoration: none;
  color: #fff;
  margin-left: 10px;
}

.gpp_accessibility .gs_button_close:hover {
  background: linear-gradient(to top, #2025A7, #000579);
}

.gpp_accessibility .bottom .submit {
  border: 1px solid #000579;
  background: #fff;
  cursor: pointer;
  color: #000579 !important;
}

.gpp_accessibility .bottom .acceptAll {
  margin-left: 14px;
  margin-right: 14px;
}

.gpp_accessibility .acceptAll,
.gpp_accessibility .declineAll {
  cursor: pointer;
  background: linear-gradient(to bottom, #2025A7, #000579);
}

.gpp_accessibility .close_button:hover,
.gpp_accessibility .bottom .submit:hover {
  background-color: #F5F9FF;
}

.gpp_accessibility .bottom .acceptAll:hover,
.gpp_accessibility .bottom .declineAll:hover {
  cursor: pointer;
  background: linear-gradient(to top, #2025A7, #000579);
}

.gpp_accessibility .bottom {
  justify-content: space-between;
  flex-direction: column;
  height: auto !important;
  margin-left: 0;
  margin-right: 0;
  gap: 14px;
}

.gpp_accessibility .prefPanel {
  margin: 0;
  padding: 0;
}

@media (max-width: 768px) {
  .gpp_accessibility .bottom .acceptAll {
    margin-left: unset !important;
    margin-right: unset !important;
  }
}

@media (min-width: 768px) {
  .gpp_accessibility .bottom {
    flex-direction: row;
  }
}

@media (max-width: 390px) {
  .gpp_accessibility .bottom .acceptAll,
  .gpp_accessibility .bottom .declineAll {
    display: none;
  }

  .gpp_accessibility .bottom .submit {
    color: #fff !important;
    background: linear-gradient(to bottom, #2025A7, #000579);
  }

  .gpp_accessibility .bottom .submit:hover {
    background: linear-gradient(to top, #2025A7, #000579);
  }
}

.gpp_accessibility .layoutLabel {
  margin-top: 14px;
  padding-bottom: 21px;
}

.gpp_accessibility .viewcookie {
  border-radius: 6px;
  display: block !important;
  align-items: center;
  color: #1E293B;
  margin-top: 0;
}

.gpp_accessibility .viewcookie a {
  cursor: pointer;
  font-size: 14px !important;
  color: #334155;
}

.gpp_accessibility .viewcookie .subsections_header {
  height: 54px;
  border-bottom: unset;
  border-top-left-radius: 6px;
  border-top-right-radius: 6px;
  display: flex;
  align-items: center;
  padding: 7px;
  background-color: #F8FAFC;
  font-weight: 700;
}

.gpp_accessibility .viewcookie .table .subsections,
.gpp_accessibility .viewcookie .subsections_header {
  border: 1px solid #E3E8F0;
}

.gpp_accessibility .viewcookie .table .subsections {
  border-top: unset;
  border-bottom-left-radius: 6px;
  border-bottom-right-radius: 6px;
  margin-bottom: 20px;
}

.gpp_accessibility .viewcookie .table .subsections > div:not(:last-child) {
  border-bottom: 1px solid #E3E8F0;
}

.gpp_accessibility .viewcookie .table .subsections > div {
  padding: 7px;
  height: 36px;
  display: flex;
  align-items: center;
}

.gpp_accessibility .header {
  font-family: inherit;
  color: #334155;
  background: #F8F9FA;
  padding: 14px;
  display: flex;
  align-items: center;
  border-radius: 6px;
  width: 100%;
  border: 1px solid #E2E8F0;
}

.gpp_accessibility .viewcookie .header div {
  font-weight: 600;
  line-height: 1.5;
  font-size: 14px;
}

.gpp_accessibility .normalArrow,
.gpp_accessibility .activeArrow {
  cursor: pointer;
  margin-right: 10px;
}

.gpp_accessibility .normalArrow {
  transform: rotate(0deg);
  transition: transform 0.2s;
}

.gpp_accessibility .activeArrow {
  transform: rotate(90deg);
  transition: transform 0.2s;
}

.gpp_accessibility .viewcookie .table {
  padding: 20px;
}

.gpp_accessibility .viewcookie .table span {
  display: block;
  line-height: 24px;
}

.gpp_accessibility .viewcookie .table .title span:first-child {
  font-weight: 700;
  margin-bottom: 7px;
}

.gpp_accessibility .prefTable {
  width: 100%;
  background: #fff;
  overflow: visible;
  border-collapse: unset;
  -webkit-border-radius: 6px;
  border-radius: 6px;
  -moz-border-radius: 6px;
  border: 1px solid #E3E8F0;
  border-spacing: 0;
  margin-top: 14px;
}

.gpp_accessibility .prefTable thead tr {
  height: unset;
}

.gpp_accessibility .prefTable thead tr th:first-child {
  border-left: unset !important;
}

.gpp_accessibility .prefTable thead tr th:last-child {
  border-right: unset !important;
}

.gpp_accessibility .prefTable thead tr:first-child th {
  font-weight: bold;
  background-color: #F8FAFC;
  padding: 7px;
  border-bottom: 1px solid #E3E8F0;
}

.gpp_accessibility .prefTable tr {
  height: 56px;
}

.gpp_accessibility .prefTable tr td {
  vertical-align: middle;
  padding: 7px;
  border-bottom: 1px solid #E3E8F0;
}

.gpp_accessibility .prefTable tr td {
  border-bottom: 1px solid #eee;
  vertical-align: middle;
  padding: 7px;
}

.gpp_accessibility .prefTable tr td .whyCantIOptOut {
  padding: 7px;
}

.gpp_accessibility .prefTable tr:last-child td {
  border-color: #ddd;
}

.gpp_accessibility .prefTable tr td:first-child {
  border-left: unset !important;
}

.gpp_accessibility .prefTable tr td:last-child {
  border-right: unset !important;
}

.gpp_accessibility .contentfield {
  margin-bottom: 25px;
  font-weight: 600;
}

.gpp_accessibility .logo {
  height: 48px;
  object-fit: contain;
}

.gpp_accessibility .whyCantIOptOut a {
  color: #1D4ED8 !important;
}

.gpp_accessibility .linksContainer {
  margin-top: 15px;
}

.gpp_accessibility .seemore {
  color: #1D4ED8 !important;
}

.gpp_accessibility .gpp_switch {
  display: flex;
  width: 70px;
  text-align: center;
}

.gpp_accessibility .gpp_category {
  border-top: 1px solid #dddd;
}

.gpp_accessibility .gpp {
  display: flex;
  flex-direction: column;
}

.gpp_accessibility .logo {
  max-width: 250px;
  margin-left: auto;
  margin-right: auto;
}

.gpp_accessibility .cookiesAndRelated {
  text-align: center;
  width: 100%;
  display: flex;
  font-weight: 500;
  margin: 0 auto;
  justify-content: center;
  font-size: 14px;
  padding-bottom: 10px;
  padding-top: 5px;
}

.gpp_accessibility .acceptAll {
  color: white !important;
}

.gpp_accessibility .declineAll {
  color: white !important;
}

.gpp_accessibility .gppPrefPanel {
  margin-top: 0 !important;
  display: flex;
  flex-direction: column;
}

.gpp_accessibility .prefPanel .left {
  width: 100%;
}

.gpp_accessibility .selectNotice {
  display: flex;
  justify-content: center;
  font-size: 14px;
  color: #1D4ED8;
  font-weight: 500;
}

.gpp_accessibility .bottom {
  display: flex;
  text-align: center;
  line-height: 3;
  text-transform: uppercase;
  margin-top: 10px;
}

.gpp_accessibility .activeArrow {
  display: flex;
}

.gpp_accessibility .normalArrow {
  display: flex;
}

.gpp_accessibility .purposeName {
  color: #3B82F6;
}

.gpp_accessibility .subsections {
  flex-direction: column;
  width: 100%;
  display: flex;
  gap: 2px;
}

.gpp_accessibility .subsectionsLabel {
  font-weight: 500;
  margin-top: 10px;
  margin-bottom: 15px;
}

.gpp_accessibility .footer {
  background: none repeat scroll 0 0 #ffffff;
  border-top: 1px solid #EEEEEE;
  font-size: 14px;
  margin-top: 20px;
  padding: 10px 20px;
  color: #999;
  overflow: hidden;
}

.gpp_accessibility .gpp_subsection {
  display: flex;
}

.gpp_accessibility .gppViewSwitchPanel {
  display: flex;
  align-items: center;
}

.gpp_accessibility .layoutLabel {
  padding-top: 10px;
}

.gpp_accessibility .subsections .gpp_switch {
  margin-left: auto;
}

.gsFooter.footerAccessibleContainer {
  display: flex;
  height: 80px !important;
}

.acccessiblePoweredBy {
  font-size: 14px;
  line-height: 1.5;
  font-weight: 600;
}

.acccessiblePoweredBy img {
  padding-left: 2px;
  height: 13px;
  width: 61px;
  padding-bottom: 2px;
  vertical-align: middle;
}

.gpp_accessibility .close_button {
  float: right;
  width: 28px;
  height: 28px;
  background-image: url("asset/close-button.svg");
  cursor: pointer;
  border-radius: 100%;
  outline: none;
}

.loyaltyMessageAccessible .loyaltyMessageHeader,
.loyaltyMessageAccessible .loyaltyMessageBody,
.loyaltyMessageAccessible .loyaltyMessageBottom {
  font-family: "Source Sans Pro", sans-serif;
}

.loyaltyMessageAccessible {
  padding: 1px 1px 0px 1px;
}

.loyaltyMessageAccessible .loyaltyMessageHeader {
  justify-content: unset;
  height: unset;
  box-shadow: none;
  position: relative;
}

.loyaltyMessageAccessible .loyaltyMessageHeader h2 {
  font-weight: 600;
  font-size: 21px;
  color: #1E293B;
  line-height: 31.5px;
}

.loyaltyMessageAccessible .loyaltyMessageBody {
  padding: 21px 0;
  font-size: 14px;
  font-weight: 600;
  color: #1E293B;
  line-height: 21px;
}

.loyaltyMessageAccessible #trustarc-internal-close-button {
  position: absolute;
  right: 0;
  top: 0;
}

.loyaltyMessageNoAccessible,
.loyaltyMessageYesAccessible {
  background: linear-gradient(360deg, #000579 -4.26%, #2025A7 97.87%);
  color: white !important;
}

.loyaltyMessageAccessible .loyaltyMessageBottom {
  box-shadow: none;
}

.p-regular {
  font-size: 14px;
  font-weight: 400;
  line-height: 1.5;
  text-align: left;
}

.p-semibold, .disableSubmitMessage {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
  text-align: left;
}

.p-bold {
  font-size: 14px;
  font-weight: 700;
  line-height: 1.5;
  text-align: left;
}

/*
 * IAB Accessibility Style Section
 */

/* --------------------
 * 1. RESET & BASE STYLES
 * -------------------- */

/* Base font styles */
.iab_accessibility * {
  font-family: "Source Sans Pro", sans-serif;
  line-height: 1.5;
  font-weight: 600;
  font-size: 14px;
  color: #1E293B;
}

/* Reset the mainContent padding and margins - with fallback for browsers without :has() */
.mainContent.iab_accessibility_container {
  padding: 0px;
  margin: 0px;
}

/* Modern browsers that support :has() */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility) {
    padding: 0px;
    margin: 0px;
  }
}

/* Typography */
.iab_accessibility .p-semibold,
.iab_accessibility .p-semibold *,
.iab_accessibility .descriptionTextSecond {
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;
}

.iab_accessibility .p-bold,
.iab_accessibility .p-bold *,
.iab_accessibility .descriptionTextFirst {
  font-size: 14px;
  font-weight: 700;
  line-height: 1.5;
}

/* --------------------
 * 2. LAYOUT COMPONENTS
 * -------------------- */

/* Container layouts */
.iab_accessibility .contentContainer {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  padding-bottom: 21px;
}

/* Main Header */
.iab_accessibility .header {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center;
  height: 80px;
  border-radius: 4px;
}

.iab_accessibility .logo {
  max-width: 242px;
  margin: 50px 24px 24px 32px;
  max-height: 48px;
  display: inline-block;
}

/* iab repop */
.iab_accessibility .tc-repop-msg {
  padding: 0px 20px;
}

/* Side Panel */
.iab_accessibility .contentContainer .tabs {
  border-right: 1px solid #DFE7EF;
  padding: 0px 20px;
}

/* Main Content Area */
.iab_accessibility .content {
  overflow-y: auto;
  max-height: 750px;
  padding: 0px 20px;
}

.iab_accessibility .content .managePanel_header {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  -ms-flex-direction: column;
  flex-direction: column;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
}

/* Cookie Category Area */
.iab_accessibility .content .cookiecat {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  -ms-flex-direction: column;
  flex-direction: column;
  gap: 7px;
  padding-bottom: 10px;
}

.iab_accessibility .content .categoryTitleContainer {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  -ms-flex-direction: row;
  flex-direction: row;
  -webkit-box-pack: justify;
  -ms-flex-pack: justify;
  justify-content: space-between;
  -webkit-box-align: start;
  -ms-flex-align: start;
  align-items: flex-start;
  gap: 20px;
}

/* Bottom Panel Area */
.iab_accessibility .bottom {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -ms-flex-pack: distribute;
  justify-content: space-around;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  gap: 14px;
  background: #fff;
  padding: 10px 20px;
  border-radius: 40px;
  -ms-flex-wrap: wrap;
  flex-wrap: wrap;
}

/* Dividers */
.iab_accessibility .divider {
  height: 7px;
  border: 0;
  padding-top: 14px;
  padding-bottom: 14px;
}

.iab_accessibility .divider > hr {
  border: none;
  border-top: 1px solid #DFE7EF;
  height: auto;
}

/* --------------------
 * 3. UI COMPONENTS
 * -------------------- */

/* Active Status */
.iab_accessibility .activeLabel {
  font-size: 14px;
  font-weight: 600;
  min-width: 120px;
  width: 120px;
  height: 40px;
  border-radius: 32px;
  border: 0;
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center;
  cursor: default;
  background: #E4F8F0;
  color: #136C34;
}

/* Switches */
.iab_accessibility .consent,
.iab_accessibility .legitint {
  padding: unset;
  margin: unset;
}

.iab_accessibility .switchContainer {
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-orient: horizontal;
  -webkit-box-direction: normal;
  -ms-flex-direction: row;
  flex-direction: row;
  white-space: nowrap;
  min-width: 120px;
  width: 120px;
  gap: 4px;
  -ms-flex-line-pack: center;
  align-content: center;
}

/* Fallback for :not(:empty) and :has() selectors */
.iab_accessibility .switchContainer.non-empty:not(.has-active) {
  border: 1px solid #64748B;
  border-radius: 40px;
  background: rgba(245, 245, 245, 1);
}

/* Modern browsers with :has() support */
@supports selector(:has(*)) {
  .iab_accessibility .switchContainer:not(:empty):not(:has(.activeLabel)) {
    border: 1px solid #64748B;
    border-radius: 40px;
    background: rgba(245, 245, 245, 1);
    width: fit-content;
    min-width: fit-content;
  }
}

.iab_accessibility .switchContainer button {
  height: 40px;
  background: #FFFFFF;
  color: #1E293B;
  margin: 0 auto;
  padding: 7px 16.5px;
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  display: inline-block;
  text-align: center;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center;
  cursor: pointer;
  font-family: inherit;
  font-size: inherit;
}

.iab_accessibility .switchContainer button:first-child {
  border-radius: 20px 0 0 20px;
  line-height: 1.5;
  font-weight: 600;
  -webkit-box-flex: 1;
  -ms-flex-positive: 1;
  flex-grow: 1;
}

.iab_accessibility .switchContainer button:last-child {
  border-radius: 0 20px 20px 0;
  line-height: 1.5;
  font-weight: 600;
  -webkit-box-flex: 1;
  -ms-flex-positive: 1;
  flex-grow: 1;
}

/* Bottom Buttons */
.iab_accessibility .bottom button {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-flex: 1;
  -ms-flex: 1;
  flex: 1;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  padding: 10px 17px;
  margin: 0;
  background-color: #000579;
  background-image: -webkit-gradient(linear, left top, left bottom, from(#2025A7), to(#000579));
  background-image: -o-linear-gradient(#2025A7, #000579);
  background-image: linear-gradient(#2025A7, #000579);
  color: #FFFFFF;
  height: 48px;
  text-decoration: none;
  border: none;
  -webkit-box-shadow: none;
  box-shadow: none;
  outline: none;
  -webkit-transition: background-color 0.3s;
  -o-transition: background-color 0.3s;
  transition: background-color 0.3s;
  border-radius: 40px;
  font-family: inherit;
  font-size: inherit;
  font-weight: 600;
  line-height: 1.5;
  cursor: pointer;
  gap: 7px;
  min-width: 250px;
  width: 100%;
  -webkit-box-flex: 0;
  -webkit-box-sizing: border-box;
  box-sizing: border-box;
  white-space: nowrap;
}

.iab_accessibility .bottom .accessibleSaveAndExit,
.iab_accessibility .bottom .goBack {
  color: #000579;
  background-color: #FFFFFF;
  background-image: none;
  border: 1px solid #000579;
}

.iab_accessibility .bottom-even button {
  min-width: 45%;
}

.iab_accessibility .bottom-odd button {
  min-width: 30%;
}

  /* Links */
.iab_accessibility a[href],
.iab_accessibility a[id] {
  color: #1D4ED8 !important;
  text-decoration: underline !important;
  border-radius: 4px;
  font-size: 14px;
  font-weight: 600;
}

/* Accordion */
.iab_accessibility .accordionContainer {
  border-radius: 0 0 6px 6px;
  overflow: visible;
}

.iab_accessibility .accordionContainer .examplePurposes * {
  font-family: Source Sans Pro, sans-serif;
  font-weight: 700;
  font-size: 14px;
  line-height: 21px;
}

/* Toggle Button */
.iab_accessibility .toggleButton {
  width: 100%;
  height: 49px;
  padding: 14px;
  color: #1E293B;
  background-color: #F8F9FA;
  border: 1px solid #E2E8F0;
  border-radius: 6px;
  text-align: left;
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-align: center;
  -ms-flex-align: center;
  align-items: center;
  cursor: pointer;
  font-family: inherit;
  font-size: inherit;
  position: relative;
  margin-top: 10px;
}

.iab_accessibility .content .toggleButton[aria-expanded="true"] {
  border-radius: 6px 6px 0 0;
}

.iab_accessibility .content .toggleButton img {
  margin-right: 3px;
  width: 14px;
  height: 14px;
}

.iab_accessibility .content .toggleButton[aria-expanded="true"] img {
  -webkit-transform: rotate(90deg);
  -ms-transform: rotate(90deg);
  transform: rotate(90deg);
}

/* Tab Container */
.iab_accessibility .tabContainer button {
  width: 200px;
  height: 42px;
  border-radius: 6px;
  color: #1E293B;
  text-align: left;
  font-weight: 600;
  cursor: pointer;
  font-family: inherit;
  font-size: inherit;
  padding: 10px;
}

.iab_accessibility .tabContainer button.selected {
  background: #EFF6FF;
  color: #1D4ED8;
}

.iab_accessibility .tabContainer button:hover {
  outline: 2px solid #3699F1;
}

/* Preference Table */
.iab_accessibility .noCookiesTable th {
  padding: 15px 10px !important;
}

.iab_accessibility .prefTable {
  border-collapse: separate;
  border-spacing: 0;
  border-radius: 6px;
  border: 1px solid #e5e7eb;
}

.iab_accessibility .prefTable thead {
  background-color: #f9fafb;
}

.iab_accessibility .prefTable thead tr:first-child th {
  background-color: #F8FAFC;
  font-weight: 700;
  padding: 10px 10px;
}

.iab_accessibility .prefTable th {
  background-color: #F8FAFC !important;
  font-weight: 700;
  white-space: nowrap;
  border: none !important;
}

.iab_accessibility .prefTable td,
.iab_accessibility .prefTable tr {
  font-size: 14px;
  border: none;
  border-top: 1px solid #e5e7eb;
}

.iab_accessibility .prefTable td:first-child {
  width: 10px;
}

.iab_accessibility .prefTable tr td:first-child,
.iab_accessibility .prefTable tr td:last-child {
  border: none;
  border-top: 1px solid #e5e7eb;
}

.iab_accessibility .prefTable tr th:first-child,
.iab_accessibility .prefTable tr th:last-child {
  border: none;
  border-top-left-radius: 6px;
  border-top-right-radius: 6px;
}

.iab_accessibility .prefTable tr td,
.iab_accessibility .prefTable tr th {
  padding: 7px;
  overflow: visible;
  word-break: break-word;
}

.iab_accessibility .prefTable td[colspan] {
  padding: 14px;
}

.iab_accessibility .prefTable td[header^="blank-col"] {
  width: 42px;
}

.iab_accessibility .prefTable td[header^="domain-col"] .hideText {
  white-space: nowrap;
  overflow: hidden;
  max-width: 70%;
}

.iab_accessibility .prefTable td[header^="domain-col"] {
  position: relative;

  .domainPanelContainer {
    width: 100%;
  }
}

.iab_accessibility .prefTable td[header^="company-col"] {
  width: 120px;
  text-overflow: ellipsis;
  text-wrap: nowrap;
  /* Fallback for browsers that don't support text-wrap */
  white-space: nowrap;
  overflow: hidden;
}

.iab_accessibility .prefTable td[header^="legitInt-col"],
.iab_accessibility .prefTable td[header^="consent-col"] {
  width: 100px;
  text-overflow: ellipsis;
  text-wrap: nowrap;
  /* Fallback for browsers that don't support text-wrap */
  white-space: nowrap;
  overflow: hidden;
}

.iab_accessibility .prefTable p {
  margin: unset;
}
.iab_accessibility .table-container .prefTable {
  table-layout: fixed;
}

.iab_accessibility .table-container .prefTable td[header^="company-col"] {
  text-wrap: unset;
  white-space: unset;
}

.iab_accessibility .table-container .prefTable col {
  width: 56px;
}

/* List Items */
.iab_accessibility .stacksAndPurposes .listItem,
.iab_accessibility .stacksAndPurposesContainer .listItem,
.iab_accessibility .primaryPurposeContainer .listItem {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-pack: justify;
  -ms-flex-pack: justify;
  justify-content: space-between;
  -webkit-box-align: start;
  -ms-flex-align: start;
  align-items: start;
}

.iab_accessibility .stacksAndPurposesContainer .listItem,
.iab_accessibility .primaryPurposeContainer .listItem {
  -webkit-box-align: baseline;
  -ms-flex-align: baseline;
  align-items: baseline;
  padding-bottom: 10px;
  border: none !important;
}

.iab_accessibility .stacksAndPurposes .categories .listItem:first-child {
  border-top: 1px solid #DFE7EF;
  padding-top: 15px;
}

.iab_accessibility .stacksAndPurposes .listItem .flex-column,
.iab_accessibility .stacksAndPurposesContainer .listItem .purposesListText {
  -webkit-box-flex: 1;
  -ms-flex: 1;
  flex: 1;
}

/* Title */
.iab_accessibility .title {
  padding: 10px 50px 10px 20px;
}

/* --------------------
 * 4. STATE STYLES
 * -------------------- */

/* Active states */
.iab_accessibility .switchContainer > button[class*="active"]:not(:disabled) {
  background: #000579;
  color: #FFFFFF;
  -webkit-box-shadow: inset 0 0 0 1px #000579, 0 0 0 1px #000579;
  box-shadow: inset 0 0 0 1px #000579, 0 0 0 1px #000579;
}

/* Hover states */
.iab_accessibility .switchContainer button:not(:disabled):hover {
  -webkit-box-shadow: inset 0 0 0 1px #000579, 0 0 0 1px #000579;
  box-shadow: inset 0 0 0 1px #000579, 0 0 0 1px #000579;
  background: rgba(208, 225, 253, 1);
  z-index: 1;
}

.iab_accessibility .switchContainer > button[class*="active"]:not(:disabled):hover {
  -webkit-box-shadow: 0 0 0 2px #3699F1;
  box-shadow: 0 0 0 2px #3699F1;
  background: #000579;
  z-index: 1;
}

.iab_accessibility a[href]:hover,
.iab_accessibility a[id]:hover {
  color: #1D4ED8 !important;
  text-decoration: none !important;
  cursor: pointer;
}

/* Modern focus-visible (with JS fallback) */
.js-focus-visible .iab_accessibility .switchContainer button.focus-visible,
.iab_accessibility .switchContainer button:focus-visible {
  outline: none;
  -webkit-box-shadow: inset 0px 0px 0px 1px #000579, 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  box-shadow: inset 0px 0px 0px 1px #000579, 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  z-index: 1;
}

.js-focus-visible .iab_accessibility .switchContainer > button[class*="active"].focus-visible,
.iab_accessibility .switchContainer > button[class*="active"]:focus-visible {
  outline: none;
  -webkit-box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  z-index: 1;
}

.js-focus-visible .iab_accessibility .bottom button.focus-visible,
.iab_accessibility .bottom button:focus-visible {
  outline: none;
  -webkit-box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1;
}

.js-focus-visible .iab_accessibility a[href].focus-visible,
.js-focus-visible .iab_accessibility a[id].focus-visible,
.iab_accessibility a[href]:focus-visible,
.iab_accessibility a[id]:focus-visible {
  outline: none;
  -webkit-box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
}

.js-focus-visible .iab_accessibility .tabContainer button.focus-visible,
.iab_accessibility .tabContainer button:focus-visible {
  outline: none;
  -webkit-box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
}

.js-focus-visible .iab_accessibility .content .toggleButton.focus-visible,
.iab_accessibility .content .toggleButton:focus-visible {
  outline: none;
  -webkit-box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  z-index: 1;
}

.js-focus-visible .iab_accessibility .prefTable td[header^="blank-col"] button.focus-visible,
.iab_accessibility .prefTable td[header^="blank-col"] button:focus-visible {
  outline: none;
  -webkit-box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
  box-shadow: 0px 0px 0px 1px #FFFFFF, 0px 0px 0px 4px #3699F1;
}

/* --------------------
 * 5. SPECIFIC PAGE STYLES & OVERRIDES
 * -------------------- */

/* Manage Panel specific styles */
.iab_accessibility .categories .accessibleHeaderContainer,
.iab_accessibility .content .accessibleHeaderContainer {
  margin-bottom: 10px;
}

.iab_accessibility .mb-10 {
  margin-bottom: 10px;
}

.iab_accessibility .content {
  padding-bottom: 50px;
}

.iab_accessibility .managePanel_titlefield .title {
  display: none;
}

.iab_accessibility .managePanel_titlefield .logo {
  max-width: 242px;
  display: block;
  margin-top: 14px;
  margin-bottom: 21px;
}

.iab_accessibility .managePanel_titlefield .logo img {
  max-height: 48px;
}

.managePanel_titlefield #trustarc-internal-close-button {
  display: none;
}

.iab_accessibility .managePanel_text1 * {
  font-size: 14px;
  font-family: Source Sans Pro;
  font-weight: 700;
  line-height: 21px;
}

.iab_accessibility .managePanel_text2 * {
  font-weight: 600;
  font-size: 14px;
  font-family: Source Sans Pro;
  line-height: 21px;
  padding-top: 7px;
}

/* Hide go back button on manage panel */
.iab_accessibility .bottom .goBack {
  display: none;
}

/* This requires a class-based fallback since :has() isn't supported in older browsers */
.iab_accessible_managePanel .bottom .goBack {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
}

/* Manage panel fallbacks */
.mainContent.iab_accessibility_container.iab_accessible_managePanel {
  padding: 0px;
  margin: 0px;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessible_managePanel):has(.iab_accessibility) {
    padding: 0px;
    margin: 0px;
  }
}

/* Landing page styles and fallbacks */
.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) a.accessibleHeaderContainer {
  font-size: 21px;
}

.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) .legintSwitchContainer {
  display: none;
}

.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) .descriptionTextSecond {
  padding-bottom: 35px;
  margin-bottom: 15px;
  padding-top: 10px;
}

.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) div[class^="descriptionText"] {
  text-align: center;
  margin: 0 auto;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) a.accessibleHeaderContainer {
    font-size: 21px;
  }

  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) .legintSwitchContainer {
    display: none;
  }

  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) .descriptionTextSecond {
    padding-bottom: 35px;
    margin-bottom: 15px;
    padding-top: 10px;
  }

  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) div[class^="descriptionText"] {
    text-align: center;
    margin: 0 auto;
  }
}

.iab_accessibility > .accessibleHeaderContainer {
  width: 100%;
}

.iab_accessibility > .accessibleHeaderContainer div {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  -ms-flex-direction: column;
  flex-direction: column;
}

/* Logo positioning fallbacks */
.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) .accessibleHeaderContainer.has-logo .logo {
  margin: 20px 24px 24px 32px;
  -ms-flex-item-align: center;
  -ms-grid-row-align: center;
  align-self: center;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) .accessibleHeaderContainer:has(.logo) .logo {
    margin: 20px 24px 24px 32px;
    -ms-flex-item-align: center;
    -ms-grid-row-align: center;
    align-self: center;
  }
}

.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) .listItem .purposesListText {
  display: list-item !important;
  margin-left: 20px;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) .listItem .purposesListText {
    display: list-item !important;
    margin-left: 20px;
  }
}

.iab_accessibility .consentInfoContainer {
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  -ms-flex-direction: column;
  flex-direction: column;
  gap: 20px;
  padding-bottom: 15px;
}

.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) .bottom {
  -webkit-box-orient: vertical;
  -webkit-box-direction: normal;
  -ms-flex-direction: column;
  flex-direction: column;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) .bottom {
    -webkit-box-orient: vertical;
    -webkit-box-direction: normal;
  }
}

/* Details followed by list item */
.iab_accessibility h3.purposeDescription {
  margin-top: 10px;
}
.iab_accessibility .purpose-description {
  margin-bottom: 10px;
  margin-top: 10px;
}

.iab_accessibility .purposeDescription li {
  margin-left: 20px;
}

.mainContent.iab_accessibility_container .accessibleHeaderContainer #trustarc-internal-close-button {
  position: absolute;
  top: 15px;
  right: 15px;
  z-index: 1;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility) .accessibleHeaderContainer #trustarc-internal-close-button {
    position: absolute;
    top: 15px;
    right: 15px;
    z-index: 1;
  }
}

.mainContent.iab_accessibility_container:not(.iab_accessible_managePanel) .logo {
  width: 100%;
  display: -webkit-box;
  display: -ms-flexbox;
  display: flex;
  -webkit-box-pack: center;
  -ms-flex-pack: center;
  justify-content: center;
  -webkit-box-ordinal-group: 4;
  -ms-flex-order: 3;
  order: 3;
  margin-top: 15px;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:has(.iab_accessibility):not(:has(.iab_accessible_managePanel)) .logo {
    width: 100%;
    display: -webkit-box;
    display: -ms-flexbox;
    display: flex;
    -webkit-box-pack: center;
    -ms-flex-pack: center;
    justify-content: center;
    -webkit-box-ordinal-group: 4;
    -ms-flex-order: 3;
    order: 3;
    margin-top: 15px;
  }
}

.mainContent:not(.iab_accessible_managePanel) .iab_accessibility .landingPage {
  padding: 0px 20px 0px 20px;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .mainContent:not(:has(.iab_accessible_managePanel)) .iab_accessibility .landingPage {
    padding: 0px 20px 0px 20px;
  }
}

/* Conditional styling for margins with list items */
.iab_accessibility .stacksAndPurposes .listItem.has-active-switch .flex-column,
.iab_accessibility .stacksAndPurposes .listItem.has-switch .flex-column,
.iab_accessibility .stacksAndPurposesContainer .listItem.has-switch .purposesListText,
.iab_accessibility .primaryPurposeContainer .listItem.has-switch .purposesListText {
  margin-right: 15px;
}

/* Modern :has() support */
@supports selector(:has(*)) {
  .iab_accessibility .stacksAndPurposes .listItem:has(span[class*="active"]) .flex-column,
  .iab_accessibility .stacksAndPurposes .listItem:has(div[class*="switch"]) .flex-column,
  .iab_accessibility .stacksAndPurposesContainer .listItem:has(div[class*="switch"]) .purposesListText,
  .iab_accessibility .primaryPurposeContainer .listItem:has(div[class*="switch"]) .purposesListText {
    margin-right: 15px;
  }
}

/* --------------------
 * 6. MEDIA QUERIES
 * -------------------- */

@media screen and (max-width: 576px) {
  .tabs {
    display: none;
  }

  .iab_accessibility .bottom {
    -webkit-box-orient: vertical;
    -webkit-box-direction: normal;
    -ms-flex-direction: column;
    flex-direction: column;
  }
}

/* --------------------
 * 7. IE11 SPECIFIC FIXES
 * -------------------- */

/* IE11 doesn't support gap in flexbox */
@media all and (-ms-high-contrast: none), (-ms-high-contrast: active) {
  /* Target IE10 and IE11 */
  .iab_accessibility .content .cookiecat > * + * {
    margin-top: 7px; /* Simulating gap */
  }

  .iab_accessibility .content .categoryTitleContainer > * + * {
    margin-left: 20px; /* Simulating gap */
  }

  .iab_accessibility .bottom > * + * {
    margin-left: 10px; /* Simulating gap */
  }

  .iab_accessibility .consentInfoContainer > * + * {
    margin-top: 20px; /* Simulating gap */
  }

  .iab_accessibility .switchContainer > * + * {
    margin-left: 2px; /* Simulating gap */
  }

  /* IE11 flex fixes */
  .iab_accessibility .contentContainer {
    display: table; /* Fallback for IE11 */
    width: 100%;
  }

  .iab_accessibility .contentContainer .tabs,
  .iab_accessibility .contentContainer .content {
    display: table-cell; /* Fallback for IE11 */
    vertical-align: top;
  }

  /* IE11 doesn't support CSS variables, ensure all colors are explicitly defined */
  .iab_accessibility .activeLabel {
    background-color: #E4F8F0;
    color: #136C34;
  }
}

/* ============================================
   Button States
   ============================================ */

/* Focus Outline */
.ta-modal-primary-btn:focus-visible,
.ta-modal-secondary-btn:focus-visible,
.ta-modal-tertiary-btn:focus-visible,
.ta-modal-primary-btn.programmatic-focus,
.ta-modal-secondary-btn.programmatic-focus,
.ta-modal-tertiary-btn.programmatic-focus,
.accessibleSwitch button:focus-visible,
.footerAccessible > a:focus-visible,
.ta-modal-accordion:focus-visible,
.accordion:focus-visible {
  outline: none;
  box-shadow: 0 0 0 1px #FFFFFF, 0 0 0 4px #3699F1;
}

/* Primary Button States */
button.ta-modal-primary-btn:hover{
  background: linear-gradient(360deg, #2025A7 -4.26%, #000579 97.87%);
  cursor: pointer;
}

button.ta-modal-primary-btn:active {
  background: #000579;
}

button.ta-modal-primary-btn:disabled {
  background: #757575;
  border-color: #757575;
  cursor: not-allowed;
}

/* Secondary Button States */
button.ta-modal-secondary-btn:not(:disabled):hover {
  background: #F5F9FF;
  cursor: pointer;
}

button.ta-modal-secondary-btn:not(:disabled):active{
  background: #D0E1FD;
}

button.ta-modal-secondary-btn:disabled,
.iab_accessibility button.ta-modal-secondary-btn:disabled{
  border: 1px solid #757575;
  color: #757575;
  cursor: not-allowed;
  background: #FFFFFF;
}

button.ta-modal-secondary-btn:disabled img {
  filter: brightness(0) saturate(100%) invert(42%) sepia(0%) saturate(0%) hue-rotate(0deg) brightness(93%) contrast(88%);
}

/* Tertiary Button States */
button.ta-modal-tertiary-btn:hover {
  background-color: #F5F9FF;
  cursor: pointer;
}

button.ta-modal-tertiary-btn:active {
  background-color: #D0E1FD;
}

/* Small Button */
.ta-modal-btn-small {
  width: 28px;
  height: 28px;
  border-radius: 100%;
}

/* Select Button States */
.accessibleSwitch button:not(:disabled):hover {
  background-color: #F5F9FF;
  -webkit-box-shadow: inset 0 0 0 1px #000579, 0 0 0 1px #000579;
  box-shadow: inset 0 0 0 1px #000579, 0 0 0 1px #000579;
}

.accessibleSwitch button.active:not(:disabled):hover {
  background-color: #000579;
  -webkit-box-shadow: 0 0 0 2px #3699F1;
  box-shadow: 0 0 0 2px #3699F1;
}

.accessibleSwitch.disable {
  border: 1px solid #757575;
}

.accessibleSwitch button:disabled,
.switchContainer button:disabled {
  cursor: not-allowed;
}

.accessibleSwitch button:not(.active):disabled,
.switchContainer button:not(.activeConsent):disabled {
  color: #757575;
  border-color: #757575;
  background: #FFFFFF;
}

.accessibleSwitch .on.active:disabled,
.accessibleSwitch .off.active:disabled,
.switchContainer .consentno.activeConsent:disabled,
.switchContainer .consentyes.activeConsent:disabled {
  box-shadow: 0 0 0 1px #757575;
  background: #757575;
  border-width: 1px 0 1px 1px;
  border-color: #757575;
  margin: 0;
  color: #FFFFFF;
}

.accessibleSwitch button:focus-visible {
  border: 1px solid #000579;
}

/* ============================================
   Accordion States
   ============================================ */

button.ta-modal-accordion:hover,
.accordion:hover {
  border: 1px solid #3699F1;
}


/* Accessibility Feedback Styles */
.ta-feedbackMessageContainer {
  margin-top: 14px;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.ta-feedback {
  padding: 10.5px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 14px;
  font-weight: 600;
  line-height: 1.5;

  /* ta-feedback-success default style*/
  color: #136C34;
  background-color: #E4F8F0;


}

.ta-feedback-error {
  color: #B32B23;
  background-color: #FFE7E6;
}

.ta-feedback-warning {
  color: #816204;
  background-color: #FFF2E2;
}

.ta-feedback-informational {
  color: #5C2F88;
  background-color: #E9E9FF;
}

.tc-warning {
  width: 14px;
  height: 14px;
  display: inline-block;
  flex-shrink: 0;
  background-image: url("asset/exclamation-circle.svg");
}

.trustarc-gpc-notice {
  display: flex;
  align-items: center;
  gap: 0.5rem;
}

/* ---- Device Storage Disclosure (IabVendorDisclosureSectionCollapsible) ---- */
.vendorDisclosuresHeader,
.vendorDomainsHeader,
.vendorSdksHeader {
  font-weight: bold;
  margin-top: 8px;
}

.vendorDisclosurePurposesRow {
  display: flex;
  gap: 8px;
}

.vendorDisclosurePurposesRow .vendorDisclosureDetailLabel {
  flex-shrink: 0;
  min-width: 120px;
}

.vendorDisclosurePurposes,
.vendorDisclosureSpecialPurposes {
  flex: 1;
  margin: 0;
  padding: 0;
  list-style: none;
}

.vendorDomains,
.vendorSdks {
  margin: 0 0 0 20px;
  padding: 0;
  list-style: none;
}

.iab_accessibility .vendorDomains,
.iab_accessibility .vendorSdks {
  margin: 0;
  padding-left: 20px;
  list-style: disc;
}

.iab_accessibility .vendorDisclosureDetailList {
  padding-left: 20px;
  margin: 0;
}

.iab_accessibility .vendorDisclosureDetailList > li,
.iab_accessibility .vendorDomains > li,
.iab_accessibility .vendorSdks > li {
  margin-left: 0;
}

.iab_accessibility .vendorDisclosureListItem {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}`,Cc=vc,Sc=`html,
body,
div,
span,
applet,
object,
iframe,
h1,
h2,
h3,
h4,
h5,
h6,
p,
blockquote,
pre,
a,
abbr,
acronym,
address,
big,
cite,
code,
del,
dfn,
em,
font,
img,
ins,
kbd,
q,
s,
samp,
small,
strike,
strong,
sub,
sup,
tt,
var,
b,
u,
i,
center,
dl,
dt,
dd,
ol,
ul,
li,
fieldset,
form,
label,
legend,
table,
caption,
tbody,
tfoot,
thead,
tr,
th,
td,
button {
  background: transparent;
  border: 0;
  margin: 0;
  padding: 0;
  vertical-align: baseline;
}

body {
  font-size: 12px;
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
  line-height: 20px;
}
body.main {
  background: url(asset/bg.png) no-repeat center 0;
  line-height: 20px;
}
body.pbg {
  background: #fff url(asset/pbg.jpg) repeat-y 1px 0;
}
input,
textarea,
select {
  font-size: 12px;
  font-family: "Lucida Grande", Arial, Helvetica, sans-serif;
}

/***INDEX.HTML***/
.mainheader {
  container-type: inline-size; /* CCM-5518, CCM-5924: make \`.gsHeader h1 -> font-size: 3cqw !important;\` works */
}
.mainHeader h1 {
  color: #2c2d31;
  font-size: 18px;
  display: inline-block;
}
.accept-decline-buttons {
  float: right;
}
#accept_all_button {
  background: no-repeat scroll 0 0 #5fc323;
  padding: 7px 10px 7px 14px;
  border: 1px solid #5fc323;
  text-shadow: none;
  color: #ffffff;
  border-radius: 4px;
  font-family: Arial, Tahoma;
  display: none;
  font-size: 14px;
  font-stretch: normal;
  font-style: normal;
  font-weight: bold;
  letter-spacing: normal;
  line-height: normal;
  text-align: center;
  text-transform: uppercase;
}
#decline_all_button {
  background: no-repeat scroll 0 0 #5fc323;
  padding: 7px 10px 7px 14px;
  border: 1px solid #5fc323;
  text-shadow: none;
  color: #ffffff;
  border-radius: 4px;
  font-family: Arial, Tahoma;
  display: none;
  font-size: 14px;
  font-stretch: normal;
  font-style: normal;
  font-weight: bold;
  letter-spacing: normal;
  line-height: normal;
  text-align: center;
  text-transform: uppercase;
}

.prefHeader {
  border-bottom: 1px solid #ddd;
  padding-bottom: 10px;
}
.prefHeader img {
  float: right;
  vertical-align: middle;
}

.mainContent {
  container-type: inline-size; /* enables container queries */
  padding: 20px 20px 0px 20px;
  position: relative;
}
.mainContent,
.consentToAll {
  line-height: 18px;
  color: #666;
  background: #fff;
  border-top: 1px solid #fff;
}
.mainContent p,
.consentToAll p {
  margin: 10px 0;
}
.mainContent a,
.consentToAll a {
  color: #069;
  text-decoration: none;
}
.mainContent button,
.consentToAll a {
  color: #069;
  text-decoration: none;
}
.mainContent a:hover,
.consentToAll a:hover {
  text-decoration: none;
  color: #fa7f06;
}
.mainTitle2 {
  margin: 0 0 20px 0;
  color: #669b00;
  font-size: 14px;
  font-family: "Abel";
  font-weight: normal;
  text-transform: uppercase;
}
.mainContent .buttons,
.consentToAll .buttons {
  margin-bottom: 0;
  margin-top: 13px;
}
.mainContent .buttons a,
.consentToAll .buttons a {
  display: inline-block;
  opacity: 0.8;
}

.mainInner {
  padding: 10px;
  border: 1px solid #ddd;
  background: #fcfcfc;
  overflow: hidden;
}
.mainInner {
  -moz-border-radius: 2px;
  -webkit-border-radius: 2px;
  border-radius: 2px;
  -moz-box-shadow: inset 0 0 3px #e7e7e7;
}

#notice p {
  font-size: 11px;
  line-height: 15px;
}
#consent {
  padding: 10px 0;
  font-size: 12px;
}
#consent p:first-child {
  margin-top: 0;
}

.consent {
  padding: 20px 0;
}
.contentfield {
  padding: 15px 0;
}

.categoryRadioButton {
  font-size: 18px;
}
.warning {
  color: red !important;
}
.warningMessage {
  padding: 7px 10px;
  background: #f9edbe;
  border: 1px solid #f0c36d;
  margin: 11px 20px 13px;
  font-size: 11px;
  line-height: 16px;
  color: #af7501;
}
.warningMessageHttps {
  margin: 11px 0px !important;
}
.contentWarningMessage {
  padding: 7px 10px;
  background: #f9edbe;
  border: 1px solid #f0c36d;
  margin: 20px 0px 0px;
  font-size: 11px;
  line-height: 16px;
  color: #af7501;
}
.repopMessage {
  padding: 7px 10px;
  background: #f9edbe;
  border: 1px solid #f0c36d;
  margin: 11px 0px 13px;
  font-size: 11px;
  line-height: 16px;
  color: #af7501;
}

/***POP UP***/
.TRUSTe_Overlay {
  position: fixed;
  left: 0;
  top: 0;
  width: 100%;
  min-height: 100%;
  background: #000;
  opacity: 0.5;
  display: none;
}
.TRUSTe_PopParent {
  position: fixed;
  top: 30%;
  left: 50%;
  display: none;
}
.TRUSTe_PopContainer {
  position: absolute;
  left: -325px;
  top: -116px;
  width: 650px;
  background: url(asset/trans.png);
  -moz-border-radius: 5px;
  -webkit-border-radius: 5px;
  border-radius: 5px;
  padding: 10px;
}
.TRUSTe_PopInnerContainer {
  /** left: -391px top: -116px; width: 800 */
  background: #fff;
  overflow: hidden;
  -moz-border-radius: 2px;
  -webkit-border-radius: 2px;
  border-radius: 2px;
}
.TRUSTe_PopIFrame {
  border: 0;
  width: 100%;
  height: 410px;
}

.leftPanel {
  float: left;
  width: 280px;
  border-right: 1px solid #ddd;
}
.rightPanel {
  margin-left: 281px;
}

.dtabmenu {
  list-style: none;
  border-bottom: 1px solid #ddd;
  overflow: hidden;
  margin-bottom: 20px;
}
.dtabmenu li {
  display: inline-block;
  width: 50%;
  float: left;
}
.dtabmenu li a {
  display: block;
  padding: 5px;
  text-align: center;
  text-transform: uppercase;
  font-weight: bold;
  font-size: 11px;
  color: #ccc;
}
.dtabmenu li a:hover {
  background: #f7f7f7;
  color: #666;
  cursor: pointer;
}
.dtabmenu li a.current {
  background: #ddd;
  color: #333;
}

/***PREF PANEL***/
.prefPanel {
  margin: 20px 0 0 0;
}
.prefPanel .left {
  float: left;
  width: 45%;
  margin-right: 5%;
  position: relative;
}
.prefPanel .right {
  float: left;
  width: 44%;
  padding-left: 30px;
  border-left: 1px solid #eee;
}

/* .prefPanel .cookiecat a.viewcookie { display: inline-block; background: url(asset/toggle.png) no-repeat center center; padding: 18px 15px; } */
/* .prefPanel .cookiecat a.viewcookie:hover { background-color: #eee; } */
/* .prefPanel .cookiecat h3 { margin-left: 30px; } */
/* .prefPanel .cookiecat span.greenbold { color: #1c9122; font-weight: bold; margin-top: 10px; } */

.prefPanel .cookiecat a.viewcookie {
  cursor: pointer;
}
.prefPanel .cookiecat button.viewcookie {
  cursor: pointer;
}
.prefPanel .cookiecat {
  padding: 10px;
  background: #fcfcfc;
  border: 1px solid #ddd;
  margin-bottom: 10px;
  -moz-border-radius: 2px;
  -webkit-border-radius: 3px;
  border-radius: 3px;
}
.prefPanel .cookiecat {
  -moz-border-radius: 2px;
  -webkit-border-radius: 2px;
  border-radius: 2px;
  -moz-box-shadow: inset 0 0 3px #e7e7e7;
}
.prefPanel .cookiecat h3 {
  margin: 0;
  font-size: 14px;
  font-weight: normal;
  color: #069;
  text-transform: uppercase;
  font-family: "Abel", "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.prefPanel .cookiecat span {
  font-size: 11px;
  font-weight: normal;
  text-transform: none;
  display: block;
  color: #666;
}
.prefPanel .cookiecat a {
  font-size: 10px;
  text-transform: uppercase;
  display: inline-block;
  margin-top: 15px;
  background: url(asset/arrow.png) no-repeat right 2px;
  padding-right: 10px;
  color: #666;
}
.prefPanel .cookiecat a.expandcookie {
  background: url(asset/arrow-off.jpg) no-repeat right 2px;
  background-size: 10px;
}
.prefPanel .cookiecat a:hover {
  cursor: pointer;
  text-decoration: underline;
}
.prefPanel .cookiecat span.greenbold {
  color: #1c9122;
  font-weight: bold;
  margin-top: 10px;
}
.prefPanel .cookiecat .loading {
  display: inline-block;
  margin-top: 15px;
  padding-right: 10px;
}

.prefPanel .cookiecatdeactivated h3 {
  margin: 0;
  font-size: 14px;
  font-weight: normal;
  color: #a9a9a9;
  text-transform: uppercase;
  font-family: "Abel", "Helvetica Neue", Helvetica, Arial, sans-serif;
}

.prefTable {
  border-collapse: collapse;
  width: 100%;
  margin-top: 20px;
  background: #fff;
  border-top: 1px solid #ddd;
}
.prefTable thead tr:first-child th {
  font-weight: bold;
  background-color: #eee;
}
.prefTable thead tr:last-child th {
  border-color: #ddd;
}
.prefTable thead tr th:first-child {
  border-left: 1px solid #ddd;
}
.prefTable thead tr th:last-child {
  border-right: 1px solid #ddd;
}
.prefTable thead tr th {
  border-bottom: 1px solid #eee;
  vertical-align: middle;
  padding: 5px;
  text-align: left;
}
.prefTable thead tr td {
  border-bottom: 1px solid #eee;
  vertical-align: middle;
  padding: 5px;
  text-align: left;
  background-color: #eee;
}
.prefTable.withoptout thead th:last-child {
  text-align: right;
}
.prefTable tr td {
  border-bottom: 1px solid #eee;
  vertical-align: middle;
  padding: 5px;
}
.prefTable tr td a {
  margin: 0 !important;
  background: none !important;
  text-transform: none !important;
  font-size: 11px !important;
  color: #069 !important;
}
.prefTable tr td .switch {
  margin: 0;
}
.prefTable tr td .linkoff {
  margin: 0;
  width: 16px;
  float: right;
}
.prefTable tr td:nth-child(3) {
  word-wrap: break-word;
  word-break: break-all;
}
.prefTable tr td:first-child {
  border-left: 1px solid #ddd;
}
.prefTable tr td:last-child {
  border-right: 1px solid #ddd;
}
.prefTable tr:last-child td {
  border-color: #ddd;
}
.prefTable .whyCantIOptOut a {
  font-size: 12px !important;
  padding-right: 0px !important;
}
.prefTable .hiddenSeeMore {
  display: none !important;
}

.collapse {
  display: inline-block;
  width: 11px;
  height: 11px;
  cursor: pointer;
}
.colplus {
  background: url(asset/plus.png) no-repeat 0 -11px;
}
.colplus:hover {
  background-position: 0 0;
}
.colminus {
  background: url(asset/minus.png) no-repeat 0 -11px;
}
.colminus:hover {
  background-position: 0 0;
}

.prefTable .whyCantIOptOut a {
  font-size: 12px !important;
  padding-right: 0px !important;
}
.hideText {
  width: 400px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.truncatedCat {
  /* hide text if it more than N lines  */
  overflow: hidden;
  /* for set '...' in absolute position */
  position: relative;
  /* use this value to count block height */
  line-height: 1em !important;
  /* max-height = line-height (1.2) * lines max number (3) */
  max-height: 7em;
  /* fix problem when last visible word doesn't adjoin right side  */
  text-align: left;
  /* place for '...' */
  margin-right: -1em;
  padding-right: 1em;
}
/* create the ... */
.truncatedCat:before {
  /* points in the end */
  content: "...";
  /* absolute position */
  position: absolute;
  /* set position to right bottom corner of block */
  right: 0;
  bottom: 0;
}
/* hide ... if we have text, which is less than or equal to max lines */
.truncatedCat:after {
  /* points in the end */
  content: "";
  /* absolute position */
  position: absolute;
  /* set position to right bottom corner of text */
  right: 0;
  /* set width and height */
  width: 1em;
  height: 1em;
  margin-top: 0.2em;
  /* bg color = bg color under block */
  background: white;
}

@media screen and (max-width: 767px) {
  .sliderWrapper {
    left: 7px !important;
  }
}

@media screen and (max-width: 630px) {
  .hideText {
    display: none;
  }
  .prefTable .hiddenSeeMore {
    display: block !important;
  }
}

@media screen and (max-width: 400px) {
  .prefTable td:nth-child(3) {
    display: none;
  }
  .optoutless td {
    border-right: 1px solid #ddd;
  }
  .prefTable td:last-child {
    display: table-cell;
  }
}

.optoutColumn {
  float: right;
}

.switch {
  float: right;
  margin: 10px 5px 0 0;
  -moz-box-shadow: 1px 1px 0 #fff;
  -webkit-box-shadow: 1px 1px 0 #fff;
  box-shadow: 1px 1px 0 #fff;
}
.switch span {
  float: left;
  display: inline-block;
  border: 1px solid #bbb;
  padding: 3px 5px;
  background: url(asset/switchbg.png) repeat-x 0 -20px;
  cursor: pointer;
}
.switch span.on {
  border-right: 0;
  -moz-border-radius: 2px 0 0 2px;
  -webkit-border-radius: 2px 0 0 2px;
  border-radius: 2px 0 0 2px;
}
.switch span.off {
  -moz-border-radius: 0 2px 2px 0;
  -webkit-border-radius: 0 2px 2px 0;
  border-radius: 0 2px 2px 0;
}
.switch span.active {
  color: #088adc;
  text-shadow: 0 0 3px #96c9fd;
  background: url(asset/switchbg.png) repeat-x 0 0;
}
.switch span.disable {
  opacity: 0.6;
  filter: alpha(opacity=60);
  cursor: default;
}

.switchtable {
  position: relative;
  float: right;
  top: 0;
  right: 0;
}

.switch {
  position: inherit;
  top: -3px;
  right: 2px;
  margin: 10px 5px 0 0;
  -moz-box-shadow: 1px 1px 0 #fff;
  -webkit-box-shadow: 1px 1px 0 #fff;
  box-shadow: 1px 1px 0 #fff;
}

.options {
  display: inline-block;
  float: left;
  margin-left: 10px;
  width: 260px;
}
.options > div {
  background: url(asset/line2.png) no-repeat 0 9px;
  padding-left: 70px;
  padding-top: 3px;
  min-height: 20px;
  min-width: 20px;
}
.options div h2 {
  font-size: 16px;
  text-transform: uppercase;
  color: #bbb;
  font-weight: normal;
  font-family: "Abel";
}
.options div p {
  font-size: 11px;
  color: #ccc;
  margin-top: 5px !important;
  line-height: 14px;
}
/**Custom Happy Studio**/
.options div.noAdvertising {
  background: white;
  padding-left: 70px;
  padding-top: 3px;
  min-height: 20px;
  min-width: 20px;
}

.options div.active h2 {
  color: #339403;
}
.options div.active p {
  color: #666;
}

/***RIGHT PANEL***/
.required,
.functional {
  display: none;
}
.required h3,
.functional h3,
.advertising h3 {
  font-size: 16px;
  font-family: "Abel";
  font-weight: normal;
  color: #007ac7;
}
.required h3.not,
.functional h3.not,
.advertising h3.not {
  color: #fa3c09;
}

.right ul {
  list-style: none;
  margin: 10px 0 0 0;
}
.right ul li {
  background: url(asset/list.png) no-repeat 0 10px;
  display: block;
  padding-left: 15px;
}
.right h3 {
  font-size: 16px;
  font-family: "Abel";
  font-weight: normal;
  color: #007ac7;
}
.right h3.not {
  color: #fa3c09;
}

.loading {
}
.loading img {
  vertical-align: middle;
}
.loading span {
  margin-left: 10px;
}
#loading2 {
  position: absolute;
  top: 40%;
  width: 95%;
  text-align: center;
  z-index: 99999;
}

/***MAIN PANEL***/
.mainPanel {
  line-height: 18px;
  color: #666;
  overflow: hidden;
}
.mainInner,
.submission,
.shortNoticePanel {
  padding: 20px;
}
.mainTitle {
  font-size: 18px;
  color: #fa7f06;
  font-weight: normal;
  margin: 15px 0 20px 0;
}
.mainPanel p,
.submission p {
  margin: 10px 0;
}
.mainPanel .small,
.submission .small {
  font-size: 11px;
  line-height: 16px;
}
.mainPanel a,
.submission a {
  color: #069;
  text-decoration: none;
}
.mainPanel a:hover,
.submission a:hover {
  color: #fa7f06;
  text-decoration: none;
}

.mainTitle3 {
  font-size: 16px;
}

/***FOOTER***/
.footer {
  background: none repeat scroll 0 0 #f7f7f7;
  border-top: 1px solid #eeeeee;
  font-size: 11px;
  margin-top: 20px;
  padding: 10px 20px;
  color: #999;
  font-size: 11px;
  overflow: hidden;
}
.footerlinetop {
  border-top: 1px solid #ddd;
  padding-top: 2px;
}
.footer p.notice {
  margin: 5px 0;
  color: #666;
  font-weight: bold;
}
.footer .right {
  float: right;
}
.footer .left {
  float: left;
}
.footer .center {
  float: left;
  display: inline-block;
  line-height: inherit;
}
.footer a {
  text-decoration: none;
}
.footer a img {
  display: inline-block;
  margin-left: 5px;
  vertical-align: middle;
}

.language-selector {
}

/***SLIDER***/
.sliderImage {
  background: url("asset/handle.jpg") no-repeat;
  height: 19px;
  width: 19px;
}
.sliderWrapper {
  padding: 0px 20px;
  display: inline-block;
  position: absolute;
  left: 16px;
}
.ui-slider-vertical {
  width: 3px;
  background: #ddd;
  -moz-box-shadow: inset 1px 1px 1px #bbb;
  -webkit-box-shadow: inset 1px 1px 1px #bbb;
  box-shadow: inset 1px 1px 1px #bbb;
  position: relative;
}
.ui-slider-vertical {
  -moz-border-radius: 50px;
  -webkit-border-radius: 50px;
  border-radius: 50px;
}
.ui-slider-vertical a {
  width: 19px;
  height: 19px;
  background: url(asset/handle.png) no-repeat 0 0;
  position: absolute;
  left: -8px;
}
.ui-slider-vertical a.ui-slider-handle {
  margin-bottom: -8px;
  outline: none;
}
.ui-slider-vertical a.ui-state-active {
}
.ui-slider-vertical .ui-slider-range {
  background: #78da48;
  width: 3px;
  position: absolute;
  left: 0;
  -moz-border-radius: 50px;
  -webkit-border-radius: 50px;
  border-radius: 50px;
}
.ui-slider-vertical .ui-slider-range {
  -moz-box-shadow: inset 1px 1px 2px #60ae3a;
  -webkit-box-shadow: inset 1px 1px 2px #60ae3a;
  box-shadow: inset 1px 1px 2px #60ae3a;
}
.ui-slider-vertical .ui-slider-range-min {
  bottom: 0;
}
.ui-slider-vertical .ui-slider-range-max {
  right: 0;
}

.draggable {
  cursor: move;
  cursor: grab;
}

.dragging {
  cursor: move;
  cursor: grabbing;
}

/***CHECK OPTIONS***/
.checkOptions {
  margin-left: 0;
  width: auto;
}
.checkOptions > div {
  padding-left: 55px;
  padding-top: 0;
  border: 1px solid #ccc;
  -moz-border-radius: 3px;
  -webkit-border-radius: 3px;
  border-radius: 3px;
  margin-bottom: 20px;
}
.checkOptions div h2 {
  padding: 10px;
  border-left: 1px solid #ddd;
  margin: 0;
}
.checkOptions div p {
  padding: 0 10px 10px 10px;
  border-left: 1px solid #ddd;
  margin: 0 !important;
}

.checkOptions div.active {
  background: #fff url(asset/radio_off.png) no-repeat 10px center;
}
.checkOptions div.active h2 {
  color: #339403;
}
.checkOptions div.active p {
  color: #666;
}

.checkOptions div.selected {
  background: #f6feeb url(asset/radio_on.png) no-repeat 10px center;
  border-color: #6bb800;
}
.checkOptions div.selected h2 {
  color: #339403;
  border-color: #6bb800;
}
.checkOptions div.selected p {
  color: #666;
  border-color: #6bb800;
}

.checkOptions div.disabled {
  background: #fcfcfc url(asset/radio_disabled.png) no-repeat 10px center;
  border-color: #ccc;
}
.checkOptions div.disabled h2 {
  color: #aaa;
  border-color: #ccc;
}
.checkOptions div.disabled p {
  color: #ccc;
  border-color: #ccc;
}
/**Custom Happy Studio**/
.checkOptions div.noAdvertisingMobile {
  background: white !important;
}

.checkOptions .highlight {
  border-left: 1px solid #ccc;
  padding-left: 10px;
  display: block;
}

/***BUTTONS***/
.pdynamicbutton {
  margin: 20px;
}
.pdynamicbutton a {
  display: inline-block;
  margin-right: 5px;
  margin-left: 5px;
  padding: 8px 15px 8px 35px;
  line-height: 15px;
  -moz-border-radius: 3px;
  -webkit-border-radius: 3px;
  border-radius: 3px;
}
.pdynamicbutton button {
  display: inline-block;
  margin-right: 5px;
  margin-left: 5px;
  padding: 8px 15px 8px 35px;
  line-height: 15px;
  -moz-border-radius: 3px;
  -webkit-border-radius: 3px;
  border-radius: 3px;
}
/*.pdynamicbutton a { -moz-box-shadow: 1px 1px 3px #a7a6a4; -webkit-box-shadow: 1px 1px 3px #a7a6a4; box-shadow: 1px 1px 3px #a7a6a4; }*/
.pdynamicbutton a {
  text-align: right;
  text-transform: uppercase;
  font-weight: bold;
  font-size: 11px;
  color: #fff;
  opacity: 0.8;
}
.pdynamicbutton button {
  text-align: right;
  text-transform: uppercase;
  font-weight: bold;
  font-size: 11px;
  color: #fff;
  opacity: 0.8;
}
.pdynamicbutton a:hover {
  text-decoration: none;
  cursor: pointer;
  color: #fff;
  opacity: 1;
}
.pdynamicbutton button:hover {
  text-decoration: none;
  cursor: pointer;
  color: #fff;
  opacity: 1;
}

.pdynamicbutton .submit {
  background: url("asset/gbutton.png") no-repeat 0 0;
  border: 1px solid #9ba74f;
  text-shadow: 1px 1px #98ab1b;
}
.pdynamicbutton .submit.disable {
  opacity: 0.6;
  filter: alpha(opacity=60);
  cursor: default;
}
.pdynamicbutton .acceptAllButtonLower {
  background: no-repeat scroll 0 0 #5fc323;
  border: 1px solid #5fc323;
  text-shadow: none;
}
.pdynamicbutton .declineAllButtonLower {
  background: no-repeat scroll 0 0 #5fc323;
  border: 1px solid #5fc323;
  text-shadow: none;
}
.pdynamicbutton .call {
  background: url("asset/gbutton.png") no-repeat 0 0;
  border: 1px solid #9ba74f;
  text-shadow: 1px 1px #98ab1b;
  float: left;
}
.pdynamicbutton .required {
  background: url("asset/gbutton_trustarc.png") no-repeat 0 0;
  border: 1px solid #9ba74f;
  text-shadow: 1px 1px #98ab1b;
  float: left;
}
.pdynamicbutton .askmelater {
  background: url("asset/obutton.png") no-repeat 0 0;
  border: 1px solid #d07d24;
  text-shadow: 1px 1px #c77d26;
  display: none;
}
.pdynamicbutton .shp {
  background: url("asset/obutton.png") no-repeat 0 0;
  border: 1px solid #d07d24;
  text-shadow: 1px 1px #c77d26;
}
.pdynamicbutton .shp2 {
  font-size: 8px !important;
  top: 20px !important;
}
.pdynamicbutton .advance {
  background: url("asset/obutton.png") no-repeat 0 0;
  border: 1px solid #d07d24;
  text-shadow: 1px 1px #c77d26;
}
.pdynamicbutton .cancel {
  background: url("asset/backbutton.png") no-repeat 0 0;
  border: 1px solid #a3a3a3;
  text-shadow: 1px 1px #adadad;
}
.pdynamicbutton .close {
  background: url("asset/graybutton.png") no-repeat 0 0;
  border: 1px solid #a3a3a3;
  text-shadow: 1px 1px #adadad;
}

.saveAndExit.disable {
  opacity: 0.6;
  filter: alpha(opacity=60);
  cursor: default;
}

.acceptallbutton {
  -moz-border-radius: 4px;
  -webkit-border-radius: 4px;
  border-radius: 4px;
  background-image: linear-gradient(to left, #5fc323, #43b400);
  color: #ffffff !important;
  font-family: Arial, Tahoma;
  cursor: pointer;
  font-size: 14px;
  font-stretch: normal;
  font-style: normal;
  float: right;
  font-weight: bold;
  letter-spacing: normal;
  line-height: normal;
  padding: 15px;
  text-align: center;
  text-shadow: none !important;
  text-transform: uppercase;
  float: right;
  margin-inline-end: 50px;
  align-self: center;
  margin-top: 10px;
}

.acceptallbuttondisabled {
  opacity: 0.6;
  cursor: not-allowed;
}

.acceptallbutton:hover:enabled {
  text-decoration: none;
  cursor: pointer;
  background-color: rgba(0, 0, 0, 0.1);
}

/***CUSTOM STYLE***/
.slimScrollDiv {
  padding-bottom: 2px;
}
.breadcrumbs {
  font-size: 10px;
  display: block;
  margin-bottom: 8px;
  color: #999;
  border-bottom: 1px dotted #eee;
}
.breadcrumbs a {
  color: #666;
}
.backprefmenu {
  cursor: pointer;
}
.notiSuccess {
  background: #f0fbdb;
  padding: 5px 10px;
  border: 1px solid #8ecd15;
  -moz-border-radius: 2px;
  -webkit-border-radius: 2px;
  border-radius: 2px;
}
.show {
  display: block;
}
.floatRight {
  float: right;
}
.noMarginRight {
  margin-right: 0;
}
.highlight {
  color: #eb831d;
  font-weight: bold;
}
#debuggerContainer {
  height: 200px;
  overflow-x: hidden;
  overflow-y: auto;
}

/***Accessibility***/
.accesibilityList {
  list-style-position: inside;
}
.accessibility {
  width: 100% !important;
}
.accessibility h2 {
  border-left: 0px !important;
}
.accessibility p {
  border-left: 0px !important;
}
.accessibility ul {
  padding-left: 30px;
}
.accessibility h3 {
  padding-left: 10px;
}
.accessibility .disabled h2 {
  text-decoration: line-through;
}
.hidden {
  display: none;
}

.disableSubmitMessage {
}

.domainTable td:nth-child(3) {
  display: block !important;
}

/*** Granular Scroll CSS **/
.gsHeader {
  position: relative;
  right: 0;
  top: 0;
  width: 100%;
  height: 65px;
  line-height: 65px;
  padding-bottom: 0px !important;
  box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.1);
  -moz-box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.1);
  -webkit-box-shadow: 0 2px 5px 0 rgba(0, 0, 0, 0.1);
  z-index: 88;
}

.gsHeader > h1 {
  margin-left: 14px;
}

.gsWarningMsg {
  position: relative;
  top: 65px !important;
  display: block;
  background-color: #e77d15;
  padding: 10px 47px 10px 15px;
  z-index: 88;
  width: 100%;
}

.gsWarningMsg div {
  color: white !important;
  margin-right: 85px;
}

.gsWarningDivider {
  width: auto;
  display: block;
  height: 1px;
  margin: 10px 40px 10px 10px !important;
  opacity: 0.25;
  border: solid 0.5px #ffffff;
}

.gsWarningMsgClose {
  color: white !important;
  cursor: pointer;
  background: url(asset/ic-close-white.svg);
  width: 9px;
  height: 9px;
  position: absolute;
  top: 10px;
  right: 80px;
}

.gsWarningBtn {
  background: url(asset/ic-warning.svg);
  display: inline-block;
  position: relative;
  margin-left: 6px;
  width: 18px;
  height: 16px;
  cursor: pointer;
}

.gsPrefPanel {
  overflow-y: auto !important;
  overflow-x: hidden !important;
  position: relative;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0 !important;
  padding: 10px 15px 0px 15px !important;
}

.gsContent {
  padding: 0px !important;
}

.gsSubmitContainer {
  width: 100% !important;
  position: relative !important;
  padding-top: 11px;
  box-shadow: 0 -5px 5px -5px rgba(0, 0, 0, 0.1);
  -moz-box-shadow: 0 -5px 5px -5px rgba(0, 0, 0, 0.1);
  -webkit-box-shadow: 0 -5px 5px -5px rgba(0, 0, 0, 0.1);
}

.gsSubmit {
  margin: 0px 15px 0px 15px !important;
  padding: 11px 15px 11px 15px !important;
  line-height: 18px !important;
  text-align: center !important;
  display: block !important;
}

.gsOptedOutMsg {
  float: left;
  margin-left: 15px;
}

div.gsOptedOutMsg .highlight {
  font-size: 17px;
  color: #004676;
}

.gsSubmitOptedOut {
  float: right;
}

.gsFooter {
  position: relative !important;
  right: 0 !important;
  bottom: 0 !important;
  width: 100% !important;
  height: 34px !important;
  padding: 0px !important;
  margin: 0px !important;
}

.gsFooter > .right,
.gsFooter > .left {
  margin-top: 6px;
}

.gsFooter > .right {
  margin-right: 15px;
}

.gsFooter > .left {
  margin-left: 15px;
}

.gsLogo {
  display: flex !important;
  float: right !important;
  height: 55px !important;
  margin-top: 5px !important;
  margin-right: 15px !important;
  position: static !important;
}

.gsLogo img {
  max-width: 160px !important;
  object-fit: contain !important;
}

.warningTooltip {
  visibility: hidden;
  left: -210%;
  position: absolute;
  top: 20px;
  text-align: center;
  width: 120px;
  line-height: 11px;
  font-size: 8px;
  z-index: 90;
  border: solid 1px #ececec;
  border-radius: 2px;
  box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.09);
  -moz-box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.09);
  -webkit-box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.09);
  background-color: #fcfcfc;
  color: #ababab;
}

.gsWarningBtn:hover .warningTooltip {
  visibility: visible !important;
}

@media only screen and (max-width: 780px) {
  .gsHeader {
    line-height: 0px;
    height: 90px;
    text-align: center;
    z-index: 888;
  }

  .gsHeader .gsLogo {
    float: none !important;
    margin-top: 10px;
    height: 35px !important;
    margin-right: 0px !important;
    justify-content: center !important;
    width: auto !important;
  }

  .gsHeader h1 {
    margin-left: 0px;
    font-size: 3cqw !important;
    margin-top: 24px !important;
    display: inline-block !important;
  }

  .gsPrefPanel {
    margin-top: 0px !important;
  }

  .gsPrefPanelSubmitted {
    margin-bottom: 130px !important;
  }

  .gsWarningMsg {
    top: 90px !important;
  }

  .gsOptedOutMsg {
    float: none;
  }

  .gsSubmitOptedOut {
    float: none;
  }

  .warningTooltip {
    left: -300%;
  }
  .accept-decline-buttons {
    float: none;
    display: block;
    margin-top: -30px;
  }

  #accept_all_button {
    display: none;
    width: 49%;
  }
  #decline_all_button {
    display: none;
    width: 49%;
  }
}

@media only screen and (max-width: 406px) {
  .gsPrefPanelSubmitted {
    margin-bottom: 147px !important;
  }

  .gsHeader h1 {
    font-size: 5cqw !important;
  }

  .gsOptedOutMsg {
    font-size: 3cqw;
  }

  div.gsOptedOutMsg .highlight {
    font-size: 4cqw;
  }

  .warningTooltip {
    left: -320%;
  }
}

.gsHeaderNoLogo {
  height: 65px;
  line-height: 65px !important;
}

.gsHeaderNoLogo h1 {
  margin-top: 0px !important;
}

.gsPrefPanelNoLogo {
}

.gsWarningMsgNoLogo {
  top: 65px !important;
}

.hiddenElement {
  visibility: hidden;
}

/*** Granular Scroll Advanced CSS ***/
.gsaHeader {
  padding: 0px !important;
}

.gsaSubmitContainer {
  margin-left: 15px;
  margin-right: 15px;
}

.gsaCancel {
  margin-left: 10px;
}

.gsaPrefPanel + .optedOutMessage {
  visibility: hidden;
}

@media only screen and (max-width: 780px) {
  .gsaLogo {
    top: 10px !important;
  }

  .gsaPrefPanel {
    margin-bottom: 194px !important;
  }
}

.advanceSettingAnchor {
  cursor: pointer;
}

.twoStep {
  margin: -20px -20px 0 -20px;
}

.twoStep .twoStepBottom .twoStepBack {
  height: 32px;

  background: #ffffff;
  border: 1px solid #555555;
  box-sizing: border-box;
  border-radius: 2px;

  color: #000000;
  cursor: pointer;
}

.twoStep .twoStepBottom .twoStepNo,
.twoStep .twoStepBottom .twoStepYes {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 32px;

  background: #1e5883;
  border-radius: 2px;

  color: #ffffff;
  cursor: pointer;
}

.twoStep .twoStepBottom .twoStepBack {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 32px;
}

.twoStepBottom {
  display: grid;
  column-gap: 2.5%;
  padding-left: 2.5%;
  padding-right: 2.5%;
  padding-top: 12px;
  grid-auto-flow: column;
  height: 44px;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 12px;
  line-height: 30px;
  /* identical to box height */
  text-align: center;

  box-shadow: 0 -1px 4px rgba(0, 0, 0, 0.1);
  background: #ffffff;
}

.twoStepHeader {
  height: 84px;
  display: flex;
  box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.07);
  justify-content: center;
  align-items: center;
}

.twoStep .twoStepBody {
  height: 175px;
  display: block;
  text-align: center;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: normal;
  font-size: 14px;
  line-height: 94px;
}

.twoStepBodyMessage {
  width: 100%;
  text-align: left;
  margin-top: 21px;
}

.twoStepBodyMessage p {
  margin: unset;
}

.switch .gwt-RadioButton {
  border: none;
  color: black;
  background-color: transparent;
  background-image: none;
}

.switch span.gwt-RadioButton.active {
  color: black;
  background-image: none;
}

.prefPanel .cookiecat button.viewcookie {
  font-size: 12px;
  font-weight: normal;
  font-style: normal;
  font-stretch: normal;
  line-height: normal;
  letter-spacing: normal;
  text-align: left;
  color: #2074e5;
  margin: 10px 0;
  padding-right: 10px;
  background-image: none;
  display: inline-block;
  text-transform: none;
}

.loyaltyMessage {
  display: flex;
  flex-direction: column;
}

.loyaltyMessage .loyaltyMessageBottom .loyaltyMessageNo {
  height: 40px;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;

  background: #1e5883;
  border-radius: 4px;

  color: #ffffff;
  cursor: pointer;
}

.loyaltyMessage .loyaltyMessageBottom .loyaltyMessageYes {
  height: 40px;
  display: flex;
  flex-direction: row;
  justify-content: center;
  align-items: center;

  background: #1e5883;
  border-radius: 4px;

  color: #ffffff;
  cursor: pointer;
}

.loyaltyMessageBottom {
  display: grid;
  column-gap: 2.5%;
  padding-left: 2.5%;
  padding-right: 2.5%;
  padding-top: 12px;
  grid-auto-flow: column;
  height: 44px;
  font-style: normal;
  font-weight: normal;
  font-size: 0.833rem;
  line-height: 30px;
  /* identical to box height */
  text-align: center;

  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.04);
}

.loyaltyMessageHeader {
  height: 98px;
  display: flex;
  box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.04);
  justify-content: center;
  align-items: center;
}

.loyaltyMessage .loyaltyMessageBody {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  padding: 24px;
  gap: 10px;

  height: 100%;

  font-family: IBM Plex Sans;
  font-style: normal;
  font-weight: 400;
  font-size: 16px;
  line-height: 22px;

  color: #000000;
  margin: auto;
}

.br-clear {
  clear: both;
}

.gppViewSwitchPanel {
  display: flex;
  align-items: center;
}

.gpp_on_switch {
  cursor: pointer;
}

.gpp_off_switch {
  cursor: pointer;
}

.bottom .submit {
  cursor: pointer;
}

.bottom .acceptAll {
  cursor: pointer;
}

.bottom .declineAll {
  cursor: pointer;
}

.viewcookie.active {
  background-color: transparent;
}

gpp_category {
  margin-bottom: 10px;
  padding: 10px;
  color: #1e293b;
  display: block;
  font-size: 16px;
}

.gpp_category > div:first-child {
  min-height: 42px;
  margin-bottom: 15px;
}

.gpp_category .category_active {
  width: 113px;
  height: 44px;
  border-radius: 40px;
  gap: 8px;
  background-color: #e4f8f0;
  color: #136c34;
  text-transform: uppercase;
  display: flex;
  float: right;
  justify-content: center;
  align-items: center;
}

.gpp_category h3 {
  font-style: normal;
  font-weight: 700;
  line-height: 36px;
  margin-top: 0;
  font-size: 24px;
}

.gpp_category > div {
  display: flex;
  justify-content: space-between;
}

.gpp_category p {
  font-weight: 500;
  line-height: 24px;
}

.gpp_category .gpp_switch {
  border-radius: 40px;
  height: 40px;
  min-width: 122px;
  align-items: center;
  flex-direction: row;
  margin-top: unset !important;
}

.gpp_category .gpp_switch span {
  margin-bottom: unset !important;
}

.gpp_category .gpp_switch .gpp_off_switch,
.gpp_category .gpp_switch .gpp_on_switch {
  align-content: center;
  height: 100%;
  line-height: 1.5;
}

.gpp_category .gpp_switch .active {
  background-color: #000579;
  color: #fff;
}

@keyframes rotate {
  0% {
    transform: rotate(0deg);
  }
  100% {
    transform: rotate(360deg);
  }
}

.gpp_category .loading {
  animation: rotate 2s linear infinite;
  display: inline-block;
  font-size: 24px;
  margin-right: 10px;
}

.gpp_category .loading.switch {
  display: flex !important;
  float: right;
  margin-right: unset !important;
}

.trustarc-gpc-notice, .trustarc-gpc-notice-gs {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.75rem;
  margin-bottom: 0.75rem;
}

.wrapper {
  background: transparent;
  position: relative;
}

.reset-html * {
  margin: 0 !important;
}

.reset-html p {
  margin-bottom: 1em !important;
}

/* GPP Loading Spinner Styles */
.gpp_loading_container {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 40px;
  min-width: 122px;
  border-radius: 40px;
  background-color: #f8f9fa;
}

.loadingSpinner {
  width: 20px;
  height: 20px;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* ---- Device Storage Disclosure (IabVendorDisclosureSectionCollapsible) ---- */
.vendorDisclosuresHeader,
.vendorDomainsHeader,
.vendorSdksHeader {
  font-weight: bold;
  margin-top: 8px;
}

.vendorDisclosurePurposesRow {
  display: flex;
  gap: 8px;
}

.vendorDisclosurePurposesRow .vendorDisclosureDetailLabel {
  flex-shrink: 0;
  min-width: 120px;
}

.vendorDisclosurePurposes,
.vendorDisclosureSpecialPurposes {
  flex: 1;
  margin: 0;
  padding: 0;
  list-style: none;
}

.vendorDomains,
.vendorSdks {
  margin: 0;
  padding: 0;
  list-style: none;
}
`,kc=Sc,Ac=function(){Tc(),Pc()};function Tc(){const n={},t=window.location.search.substring(1).split("&");for(const i of t){const o=i.split("="),a=decodeURIComponent(o[0]),r=decodeURIComponent(o[1]||"");typeof n[a]>"u"?n[a]=r:typeof n[a]=="string"?n[a]=[n[a],r]:n[a].push(r)}window.QueryString=n}function Pc(){window.loadjscssfile=function(e,t,i){let o;t==="js"?(o=document.createElement("script"),o.setAttribute("type","text/javascript"),o.setAttribute("src",e),o.setAttribute("defer",""),i&&(o.onload=i,o.onreadystatechange=i)):t==="css"&&(o=document.createElement("link"),o.setAttribute("rel","stylesheet"),o.setAttribute("type","text/css"),o.setAttribute("href",e)),o&&document.head.appendChild(o)};const n=JSON.stringify({source:"preference_manager",message:"cm_loading"});if(window.parent&&window.postMessage)window.parent.postMessage(n,"*");else{const e=window.location.href.match(/\burl=([^?&#;]+)/);e!=null&&e[1]&&(window.parent.location=decodeURIComponent(e[1]).replace(/#.*/,"")+"#"+n)}}function Ec(n){const e=navigator.userAgent||navigator.vendor||window.opera,t=/Mobile|iP(hone|od|ad)|Android|BlackBerry|IEMobile|Kindle|NetFront|Silk-Accelerated|(hpw|web)OS|Fennec|Minimo|Opera M(obi|ini)|Blazer|Dolfin|Dolphin|Skyfire|Zune/i,i=/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i;return t.test(e)||i.test(e.substring(0,4))}function Lc(n,e){let t=(n==null?void 0:n.toLowerCase())==="mobile";return t||(t=Ec()),t}const Oc=({rootProps:n,standAloneMode:e=!0})=>{const t=R(null),i=rn(),o=y(_=>_.configSlice.loaded),a=y(_=>_.configSlice.config),r=y(_=>_.configSlice.loading),l=y(_=>_.configSlice.error),c=y(_=>_.globalStateSlice.shouldShowDisabledSubmitMessage),d=y(_=>_.globalStateSlice.effectiveLocale),p=Zn(),u=y(_=>_.globalStateSlice.currentView);Wl(!0),W(()=>{Ac(),i(Oa(Lc(n.ostype)))},[]),W(()=>{e&&Ro(i,{rootProps:n,effectiveLocale:n.locale})},[i,n]),W(()=>{a.categories.length>0&&vr({cmConfig:a,cmType:n.cmType,dispatch:i})},[a.categories,a.config.defaultCategory]);const f=It(n);if(yc({wrapperRef:t,appStyles:f?Cc:kc,cmType:"ADV",userCss:a.contents.css_key,deps:[o,a.contents.css_key,n,t]}),W(()=>{e&&o&&!u&&i(Ea(Hs(n,a.contents)))},[o,u]),wc(t,a.contents.configuration_on_load_script_key),W(()=>{i(La(n)),i(oo(n.locale))},[n]),W(()=>{!r&&!l&&a.contents&&e&&(T.info("[CCM] Adv CM fully loaded, clearing partial load timeout"),i(xt()),i(Na()))},[r,l,a.contents,e,i]),W(()=>()=>{i(Ie()),i(xt())},[i]),r)return s(Mo,{});if(l)return s("div",{className:"error",children:["Error loading consent manager: ",l]});if(!r&&!l&&(!a.config||!a.categories))return s("div",{className:"error",children:"Error: Configuration not loaded properly"});function x(){let _=ja(n)?"accessibleMainContent":"mainContent";return p&&(_+=" gsContent"),_}function b(){return u==="optoutDoneMessage"?f?"optOutDoneAccessible":void 0:n.layout}function h(){let _;return Nn(n)&&(u==="euDisclosurePanel"?_="disclosureAccessible":_="prefTableAccessible"),_}const m=yl(u,Pn(n));return s(E,{children:s("div",{lang:d,children:[s("div",{className:x(),ref:t,...f&&{role:"main"},...m&&{"aria-labelledby":m},children:Bn(n)&&!Nt(u)?s("div",{className:b(),children:[s(pt,{}),s(fe,{}),s(Fe,{}),s(dt,{routes:ct})]}):Pn(n)?s("div",{className:h(),children:[s(pt,{}),u!=="euDisclosurePanel"&&s(fe,{}),s(Fe,{}),s("div",{"aria-hidden":"true",style:"display: none;"}),s(dt,{routes:ct})]}):s(E,{children:[s(pt,{}),s(Fe,{}),s("div",{"aria-hidden":"true",style:"display: none;"}),s(dt,{routes:ct})]})}),s(Gl,{}),!f&&c&&s(Vl,{})]})},"consent-manager-root")};class Dc extends Cn{constructor(e){super(e),this.state={hasError:!1,error:null}}static getDerivedStateFromError(e){return{hasError:!0,error:e}}componentDidCatch(e,t){T.error("[CCM] Uncaught error in component tree, closing modal: {}",e),T.error("[CCM] Error info: {}",t),En()}render(){return this.state.hasError?s(Mo,{}):this.props.children}}const Fc=Ji((n,e)=>{const{state:t,dispatch:i}=ss();return Et(e,()=>({getState:()=>t,dispatch:o=>{i(o)}}),[t,i]),s(E,{children:n.children},"redux-expose")});function Nc(n){if(typeof window>"u")return;const e=window.TrustArcConsentManager;window.TrustArcConsentManager=Object.assign(e??{},{renderDirectEmbed:n})}function Ic(n){n.classList.contains("trustarc_newcm_container")||n.classList.add("trustarc_newcm_container");let e=n.shadowRoot;e??(e=n.attachShadow({mode:"open"})),e.innerHTML="";const t=document.createElement("div");return t.className="wrapper",t.dataset.lenisPrevent="",t.tabIndex=-1,requestAnimationFrame(()=>t.focus()),e.appendChild(t),{shadowRoot:e,wrapperDiv:t}}const Ti="data-no-consent";function Rc(n){co()?n.removeAttribute(Ti):n.setAttribute(Ti,"true")}function pn(n){if(n!=null)return n==="false"||n==="0"?!1:!!n}function Mc(n){n.site?n.cmId=n.site:T.warn("[CCM] 'site' parameter is missing, CM will not be rendered."),n.action&&(n.action=n.action.toLowerCase()),n.userType&&(n.userType=n.userType.toUpperCase()),n.locale||(T.warn("[CCM] 'locale' parameter is missing, use default locale {}.",Ci),n.locale=Ci),n.debug=pn(n.debug),n.repop=pn(n.repop),n.layout?n.layout=n.layout.toLowerCase():n.layout="default_eu",n.type||T.warn("[CCM] Advanced CM detected but 'type' parameter is missing"),n.autoblock=pn(n.autoblock),n.accessibility=pn(n.accessibility),n.iab=pn(n.iab),n.atp=pn(n.atp),n.twoStep=pn(n.twoStep),n.optoutClose=pn(n.optoutClose),n.is_cpra=pn(n.is_cpra),n.is_finprog=pn(n.is_finprog),n.is_gpc=pn(n.is_gpc),n.is_dnt=pn(n.is_dnt),n.showGPPCategories=pn(n.showGPPCategories),n.irm=pn(n.irm),n.preview=pn(n.preview)}const Bc=({rootProps:n})=>s(Dc,{children:s(rs,{children:s(Fc,{children:s(Oc,{rootProps:n})})})});function Hc(n,e){const{wrapperDiv:t}=Ic(e);Rc(e);const i=pe();return ee(s(Bc,{rootProps:n}),t),i}function zc(n,e){T.info("[CCM] Rendering ADV CM with CM ID {}, parameters: {}",e.site,e),Mc(e);const t=Hc(e,n),i=()=>{var a;return(a=t.current)==null?void 0:a.getState()},o=a=>{var r;return(r=t.current)==null?void 0:r.dispatch(a)};return typeof window<"u"&&window.TrustArcConsentManager&&(window.TrustArcConsentManager.getState=i,window.TrustArcConsentManager.dispatch=o),{getState:i,dispatch:o}}Nc(zc);
