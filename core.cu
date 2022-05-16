struct Vec5
{
	float r = 0.0, u = 0.0, v = 0.0, p = 0.0, T = 0.0;

	__device__ Vec5 operator-(Vec5 other)
	{
		Vec5 res;
		res.r = this->r - other.r;
		res.u = this->u - other.u;
		res.v = this->v - other.v;
		res.p = this->p - other.p;
		res.T = this->T - other.T;
		return res;
	}

	__device__ Vec5 operator+(Vec5 other)
	{
		Vec5 res;
		res.r = this->r + other.r;
		res.u = this->u + other.u;
		res.v = this->v + other.v;
		res.p = this->p + other.p;
		res.T = this->T + other.T;
		return res;
	}

	__device__ Vec5 operator*(float d)
	{
		Vec5 res;
		res.r = this->r * d;
		res.u = this->u * d;
		res.v = this->v * d;
		res.p = this->p * d;
		res.T = this->T * d;
		return res;
	}
};

struct Vec4
{
	float a1 = 0.0, a2 = 0.0, a3 = 0.0, a4 = 0.0;

	__device__ Vec4 operator-(Vec4 other)
	{
		Vec4 res;
		res.a1 = this->a1 - other.a1;
		res.a2 = this->a2 - other.a2;
		res.a3 = this->a3 - other.a3;
		res.a4 = this->a4 - other.a4;
		return res;
	}

	__device__ Vec4 operator+(Vec4 other)
	{
		Vec4 res;
		res.a1 = this->a1 + other.a1;
		res.a2 = this->a2 + other.a2;
		res.a3 = this->a3 + other.a3;
		res.a4 = this->a4 + other.a4;
		return res;
	}

	__device__ Vec4 operator*(float d)
	{
		Vec4 res;
		res.a1 = this->a1 * d;
		res.a2 = this->a2 * d;
		res.a3 = this->a3 * d;
		res.a4 = this->a4 * d;
		return res;
	}

	__device__ Vec4 operator/(float d)
	{
		Vec4 res;
		res.a1 = this->a1 / d;
		res.a2 = this->a2 / d;
		res.a3 = this->a3 / d;
		res.a4 = this->a4 / d;
		return res;
	}
};

struct Vec2
{
	float x = 0.0, y = 0.0;

	__device__ Vec2 operator-(Vec2 other)
	{
		Vec2 res;
		res.x = this->x - other.x;
		res.y = this->y - other.y;
		return res;
	}

	__device__ Vec2 operator+(Vec2 other)
	{
		Vec2 res;
		res.x = this->x + other.x;
		res.y = this->y + other.y;
		return res;
	}

	__device__ Vec2 operator*(float d)
	{
		Vec2 res;
		res.x = this->x * d;
		res.y = this->y * d;
		return res;
	}

	__device__ Vec2 operator/(float d)
	{
		Vec2 res;
		res.x = this->x / d;
		res.y = this->y / d;
		return res;
	}
};

struct Color3f
{
	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	__host__ __device__ Color3f operator+ (Color3f other)
	{
		Color3f res;
		res.R = this->R + other.R;
		res.G = this->G + other.G;
		res.B = this->B + other.B;
		return res;
	}

	__host__ __device__ Color3f operator* (float d)
	{
		Color3f res;
		res.R = this->R * d;
		res.G = this->G * d;
		res.B = this->B * d;
		return res;
	}
};

static Vec5* EmptyVec5Field;

static Vec5* devVec5FieldOld;
static Vec5* devVec5FieldNew;

static Vec4* EVec4Field;
static Vec4* FVec4Field;
static Vec4* UVec4Field;

static uint8_t* ColorField;

static float* dye;

__device__ float calcE(float T, float ro, Config Con){
	float dV = Con.dx * Con.dx;
	return Con.Cv*T/ro/dV - ro*Con.a/Con.MolarMass/Con.MolarMass;
}

__device__ float calcT(float ro, float e, Config Con){
	float dV = Con.dx * Con.dx;
	return dV*ro/Con.Cv*(e+ro/Con.a/Con.MolarMass/Con.MolarMass);
	//return Con.T;
}

__device__ float calcP(float ro, float e, Config Con){
	float dV = Con.dx * Con.dx;
	return Con.R*calcT(ro, e, Con)/(dV-Con.b*ro*dV/Con.MolarMass) - Con.a*ro*ro/Con.MolarMass/Con.MolarMass;
	//return Con.P;
}

__device__ int takeIndex(int x, int y){
	return y*FIELDWIDTH + x;
}

__global__ void pre_calc_field(Vec5 *fieldOld, Vec5 *fieldNew, Vec4 *E, Vec4 *F, Vec4 *U, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;
	float dV = Con.dx*Con.dx;
	if((x<FIELDHEIGHT) && (y<FIELDHEIGHT)){
		Vec4 Un;
		Vec4 En;
		Vec4 Fn;

		Vec5 N;

		N.r = fieldOld[index].r;
		N.u = fieldOld[index].u;
		N.v = fieldOld[index].v;
		N.p = fieldOld[index].p;
		N.T = fieldOld[index].T;

		float dux = (fieldOld[takeIndex(x+1, y)].u - fieldOld[takeIndex(x-1, y)].u) / (2*Con.dx);
		float dvy = (fieldOld[takeIndex(x, y+1)].v - fieldOld[takeIndex(x, y-1)].v) / (2*Con.dx);

		float duy = (fieldOld[takeIndex(x, y+1)].u - fieldOld[takeIndex(x, y-1)].u) / (2*Con.dx);
		float dvx = (fieldOld[takeIndex(x+1, y)].v - fieldOld[takeIndex(x-1, y)].v) / (2*Con.dx);

		float TauXX = 2/3 * Con.mu * (2 * dux - dvy);
		float TauYY = 2/3 * Con.mu * (2 * dvy - dux);
		float TauXY = 2/3 * Con.mu * (2 * duy - dvx);

		float qx = -Con.TempKoef*(fieldOld[takeIndex(x+1, y)].T - fieldOld[takeIndex(x-1, y)].T) / (2*Con.dx);
		float qy = -Con.TempKoef*(fieldOld[takeIndex(x, y+1)].T - fieldOld[takeIndex(x, y-1)].T) / (2*Con.dx);

		Un.a1 = N.r;
		Un.a2 = N.r*N.u;
		Un.a3 = N.r*N.v;
		Un.a4 = (calcE(N.T, N.r, Con) + (N.u*N.u+N.v*N.v)/2)*N.r;

		En.a1 = Un.a2;
		En.a2 = Un.a2*N.u + N.p - TauXX;
		En.a3 = Un.a2*N.v - TauXY;
		En.a4 = (Un.a4 + N.p)*N.v - N.u*TauXX - N.v*TauXY + qx;

		Fn.a1 = Un.a3;
		Fn.a2 = En.a3;
		Fn.a3 = Un.a3*N.v + N.p - N.u*TauYY;
		Fn.a4 = (Un.a4 + N.p)*N.v - N.u*TauXY - N.v*TauYY + qy;

		U[index] = Un;
		E[index] = En;
		F[index] = Fn;
	}
}

__global__ void calc_field(Vec5 *fieldOld, Vec5 *fieldNew, Vec4 *E, Vec4 *F, Vec4 *U, Config Con, float dt, Vec2 force, Vec2 ForcePoint, float ForceRadius){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;
	if((x<FIELDHEIGHT) && (y<FIELDHEIGHT)){
		Vec4 Force;
		Vec4 Visc;

		if((x-ForcePoint.x)*(x-ForcePoint.x)+(y-ForcePoint.y)*(y-ForcePoint.y) <= ForceRadius){
			Force.a1 = 0;
			Force.a2 = fieldOld[index].r * force.x * dt;
			Force.a3 = fieldOld[index].r * force.y * dt;
			Force.a4 = 0;
		}else{
			Force.a1 = 0;
			Force.a2 = 0;
			Force.a3 = 0;
			Force.a4 = 0;
		}

		Visc.a1 = 0;
		Visc.a2 = ((fieldOld[takeIndex(x+1,y)].u - 2*fieldOld[index].u + fieldOld[takeIndex(x-1,y)].u) + (fieldOld[takeIndex(x,y+1)].u - 2*fieldOld[index].u + fieldOld[takeIndex(x,y-1)].u)) / (Con.dx*Con.dx);
		Visc.a3 = ((fieldOld[takeIndex(x+1,y)].v - 2*fieldOld[index].v + fieldOld[takeIndex(x-1,y)].v) + (fieldOld[takeIndex(x,y+1)].v - 2*fieldOld[index].v + fieldOld[takeIndex(x,y-1)].v)) / (Con.dx*Con.dx);
		Visc.a4 = 0;

		Vec4 Rec = ((E[takeIndex(x-1,y)] - E[takeIndex(x+1,y)]) + (F[takeIndex(x,y-1)]-F[takeIndex(x,y+1)])) * dt/(2*Con.dx) + Force + Visc * Con.nu * dt * fieldOld[index].r;
		Vec5 New;

		New.r = Rec.a1;
		New.u = Rec.a2 / Rec.a1;
		New.v = Rec.a3 / Rec.a1;
		float e = Rec.a4/ Rec.a1 - (New.u*New.u+ New.v*New.v)/2;
		New.p = calcP(Rec.a1, e, Con);
		New.T = calcT(Rec.a1, e, Con);

		fieldNew[index] = fieldOld[index] - New;
	}
}

__global__ void initField(float StartPress, float StartTemp, float StartRo, Vec5 *field){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y*FIELDWIDTH + x;

	if((x<=FIELDHEIGHT) && (y<=FIELDHEIGHT)){
		//if(((x == 0)||(x == FIELDHEIGHT - 1))&&((y == 0)||(y == FIELDWIDTH - 1))){
			field[index].p = StartPress;
			field[index].T = StartTemp;
			field[index].r = StartRo;
			field[index].u = 0.00004*index;
			field[index].v = 0.00004*index;
		//}
	}
}

__global__ void bord_cond(Vec5 *field, float *dye, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y*FIELDWIDTH + x;

	if(x == 0 && y < FIELDHEIGHT && y > 0){
		field[index].p = field[takeIndex(x+1,y)].p;
		field[index].T = field[takeIndex(x+1,y)].T;
		field[index].r = field[takeIndex(x+1,y)].r;
		field[index].u = -field[takeIndex(x+1,y)].u;
		field[index].v = field[takeIndex(x+1,y)].v;
		dye[index] = dye[takeIndex(x+1,y)];
	}

	if(x == FIELDWIDTH-1 && y < FIELDHEIGHT && y > 0){
		field[index].p = field[takeIndex(x-1,y)].p;
		field[index].T = field[takeIndex(x-1,y)].T;
		field[index].r = field[takeIndex(x-1,y)].r;
		field[index].u = -field[takeIndex(x-1,y)].u;
		field[index].v = field[takeIndex(x-1,y)].v;
		dye[index] = dye[takeIndex(x-1,y)];
	}

	if(y == 0 && x < FIELDWIDTH && x > 0){
		field[index].p = field[takeIndex(x,y+1)].p;
		field[index].T = field[takeIndex(x,y+1)].T;
		field[index].r = field[takeIndex(x,y+1)].r;
		field[index].u = field[takeIndex(x,y+1)].u;
		field[index].v = -field[takeIndex(x,y+1)].v;
		dye[index] = dye[takeIndex(x,y+1)];
	}

	if(y == FIELDHEIGHT-1 && x < FIELDWIDTH && x > 0){
		field[index].p = field[takeIndex(x,y-1)].p;
		field[index].T = field[takeIndex(x,y-1)].T;
		field[index].r = field[takeIndex(x,y-1)].r;
		field[index].u = field[takeIndex(x,y-1)].u;
		field[index].v = -field[takeIndex(x,y-1)].v;
		dye[index] = dye[takeIndex(x,y-1)];
	}

	field[index].p = Con.P;
	field[index].T = Con.T;
	field[index].r = Con.ro;

}

void init_cuda(){
    //Vec5* result;
    int FieldWidth = FIELDWIDTH;
    int FieldHeight = FIELDHEIGHT;

	cudaMalloc((void**)&devVec5FieldOld, FieldHeight*FieldWidth*sizeof(Vec5));
	cudaMalloc((void**)&devVec5FieldNew, FieldHeight*FieldWidth*sizeof(Vec5));
	cudaMalloc((void**)&EVec4Field, FieldHeight*FieldWidth*sizeof(Vec4));
	cudaMalloc((void**)&FVec4Field, FieldHeight*FieldWidth*sizeof(Vec4));
	cudaMalloc((void**)&UVec4Field, FieldHeight*FieldWidth*sizeof(Vec4)); 
	cudaMalloc((void**)&ColorField, 4*FieldHeight*FieldWidth*sizeof(uint8_t)); 
	cudaMalloc((void**)&dye, FieldHeight*FieldWidth*sizeof(float)); 

	EmptyVec5Field = (Vec5*)malloc(FieldHeight*FieldWidth*sizeof(Vec5));
	//result = (Vec5*)malloc(FieldHeight*FieldWidth*sizeof(Vec5));

	cudaMemset(&devVec5FieldNew, 0, FieldHeight*FieldWidth*sizeof(Vec5));
	cudaMemset(&devVec5FieldOld, 0, FieldHeight*FieldWidth*sizeof(Vec5));
	cudaMemset(&EVec4Field, 0, FieldHeight*FieldWidth*sizeof(Vec4));
	cudaMemset(&FVec4Field, 0, FieldHeight*FieldWidth*sizeof(Vec4));
	cudaMemset(&UVec4Field, 0, FieldHeight*FieldWidth*sizeof(Vec4));
	cudaMemset(&dye, 0, FieldHeight*FieldWidth*sizeof(float));


	// копируем ввод на device

	//cudaMemcpy( devVec5Field, EmptyVec5Field, FieldHeight*FieldWidth*sizeof(Vec5), cudaMemcpyHostToDevice);
	//cudaMemcpy( devVec5Field2, EmptyVec5Field, FieldHeight*FieldWidth*sizeof(Vec5), cudaMemcpyHostToDevice);

	//calc_field<<<1,60>>>(devVec5FieldNew);

	//cudaMemcpy(result, devVec5FieldNew, FieldHeight*FieldWidth*sizeof(Vec5), cudaMemcpyDeviceToHost);
	//std::cout<<result[0].x<<" "<<result[1].x<<std::endl;
	//cudaMemcpy(result, devVec5Field2, FieldHeight*FieldWidth*sizeof(Vec5), cudaMemcpyDeviceToHost);
	//std::cout<<result[0].x<<" "<<result[1].x<<std::endl;
	//free(result);

	dim3 blocks((FIELDWIDTH)/THREDS_PER_BLOCK+1, (FIELDHEIGHT)/THREDS_PER_BLOCK+1);
	dim3 threds(THREDS_PER_BLOCK,THREDS_PER_BLOCK);
	initField<<<blocks, threds>>>(config.P, config.T, config.ro, devVec5FieldOld);
	initField<<<blocks, threds>>>(config.P, config.T, config.ro, devVec5FieldNew);
}

__global__ void calc_dye(Vec5 *field, float *dye, Vec2 Point, float dt, float Radius, Config Con){
	int x = blockIdx.x * blockDim.x + threadIdx.x+1;
	int y = blockIdx.y * blockDim.y + threadIdx.y+1;
	int index = y*FIELDWIDTH + x;

	if((x<FIELDHEIGHT) && (y<FIELDHEIGHT)){
		float ddx = field[takeIndex(x-1,y)].u - field[takeIndex(x+1,y)].u;
		float ddy = field[takeIndex(x,y-1)].v - field[takeIndex(x,y+1)].v;

		float ddiffx = ((dye[takeIndex(x+1,y)] - 2*dye[index] + dye[takeIndex(x-1,y)]) + (dye[takeIndex(x,y+1)] - 2*dye[index] + dye[takeIndex(x,y-1)])) / (Con.dx*Con.dx);
		float ddiffy = ((dye[takeIndex(x+1,y)] - 2*dye[index] + dye[takeIndex(x-1,y)]) + (dye[takeIndex(x,y+1)] - 2*dye[index] + dye[takeIndex(x,y-1)])) / (Con.dx*Con.dx);

		float S = 0;

		if((x-Point.x)*(x-Point.x)+(y-Point.y)*(y-Point.y) <= Radius){
			S = Con.S;
		}

		dye[index] = ((ddx+ddy) * dye[index] + Con.DiffCoef*(ddiffx+ddiffy) + S)*dt;
	}
}

__global__ void PaintImage(float *field, uint8_t *res){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y*FIELDWIDTH + x;
	if((x<FIELDHEIGHT) && (y<FIELDHEIGHT)){
		//if(field[index].u != 0 || field[index].v != 0){
			res[4 * index+0] = field[index];
			res[4 * index+1] = 0;
			res[4 * index+2] = 0;
			res[4 * index+3] = 255;
		//}else{
		//	res[4 * index+0] = 0;
		//	res[4 * index+1] = 0;
		//	res[4 * index+2] = 0;
		//	res[4 * index+3] = 255;
		//}
	}
}

__global__ void print(Vec5 *field, int x, int y){
	printf("%f \n", field[y*FIELDWIDTH + x].u);
}

void RenderImage(uint8_t* result, float dt, Vec2 force, Vec2 ForcePoint, float ForceRadius){
	print<<<1,1>>>(devVec5FieldNew, 0, 1);
	//ForcePoint.x  = 1;
	//ForcePoint.y  = 1;


	dim3 blocks_calc((FIELDWIDTH-2)/THREDS_PER_BLOCK+1, (FIELDHEIGHT-2)/THREDS_PER_BLOCK+1);
	dim3 blocks_rend((FIELDWIDTH)/(THREDS_PER_BLOCK)+1, (FIELDHEIGHT)/(THREDS_PER_BLOCK)+1);
	dim3 threds_calc(THREDS_PER_BLOCK,THREDS_PER_BLOCK);
	dim3 threds_rend(THREDS_PER_BLOCK,THREDS_PER_BLOCK);

	float* chek;
	chek = (float*)malloc(FIELDHEIGHT*FIELDWIDTH*sizeof(float));

	pre_calc_field<<<blocks_calc, threds_calc>>>(devVec5FieldOld, devVec5FieldNew, EVec4Field, FVec4Field, UVec4Field, config);
	calc_field<<<blocks_calc, threds_calc>>>(devVec5FieldOld, devVec5FieldNew, EVec4Field, FVec4Field, UVec4Field, config, dt, force, ForcePoint, ForceRadius);
	
	calc_dye<<<blocks_rend, threds_rend>>>(devVec5FieldNew, dye, ForcePoint, dt, ForceRadius, config);

	bord_cond<<<blocks_rend, threds_rend>>>(devVec5FieldNew, dye, config);

	PaintImage<<<blocks_rend, threds_rend>>>(dye, ColorField);

	//cudaMemcpy(chek, dye, FIELDHEIGHT*FIELDWIDTH*sizeof(float), cudaMemcpyDeviceToHost);
	//std::cout<<chek[1000]<<std::endl;

	cudaMemcpy(EmptyVec5Field, devVec5FieldNew, FIELDHEIGHT*FIELDWIDTH*sizeof(Vec5), cudaMemcpyDeviceToHost);
	//std::cout<<EmptyVec5Field[1000].p<<" "<<EmptyVec5Field[1001].p<<std::endl;
	//std::cout<<dye[1000]<<" "<<dye[1001]<<std::endl;

	

	cudaMemcpy(result, ColorField, 4*FIELDHEIGHT*FIELDWIDTH*sizeof(uint8_t), cudaMemcpyDeviceToHost);

	std::swap(devVec5FieldOld, devVec5FieldNew);
}

void Exit_cuda(){
	free(EmptyVec5Field);

	cudaFree(devVec5FieldNew);
	cudaFree(devVec5FieldOld);
	cudaFree(EVec4Field);
	cudaFree(FVec4Field);
	cudaFree(UVec4Field);
	cudaFree(ColorField);
}