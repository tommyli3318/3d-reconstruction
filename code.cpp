OpenGLtexture zed,nx,ny,nz; // height map,normal maps (just 2D images)
picture pic;                // source image

int x,y,a;
// resize textures to source image size
zed.resize(pic.xs,pic.ys); 
 nx.resize(pic.xs,pic.ys); float *pnx=(float*) nx.txr;
 ny.resize(pic.xs,pic.ys); float *pny=(float*) ny.txr;
 nz.resize(pic.xs,pic.ys); float *pnz=(float*) nz.txr;
// prepare tmp image for height map extraction
picture pic0;
pic0=pic;       // copy
pic0.rgb2i();   // grayscale

// this computes the point cloud (this is the only important stuff from this code)
// as you can see there are just 3 lines of code important from all of this
for (a=0,y=0;y<pic.ys;y++)
 for (x=0;x<pic.xs;x++,a++)
  zed.txr[a]=pic0.p[y][x].dd>>3; // height = intensity/(2^3)

// compute normals (for OpenGL rendering only)
double n[3],p0[3],px[3],py[3];
int zedx,zedy,picx,picy;
for (a=zed.xs,zedy=-(pic.ys>>1),picy=1;picy<pic.ys;picy++,zedy++)
 for (a++,    zedx=-(pic.xs>>1),picx=1;picx<pic.xs;picx++,zedx++,a++)
    {
    vector_ld(p0,zedx-1,zedy  ,-zed.txr[a       -1]); // 3 neighboring points
    vector_ld(py,zedx  ,zedy-1,-zed.txr[a+zed.xs  ]);
    vector_ld(px,zedx  ,zedy  ,-zed.txr[a         ]);
    vector_sub(px,p0,px); // 2 vectors (latices of quad/triangle)
    vector_sub(py,p0,py);
    vector_mul(n,px,py); // cross product
    vector_one(n,n); // unit vector normalization
    pnx[a]=n[0]; // store vector components to textures
    pny[a]=n[1];
    pnz[a]=n[2];
    }